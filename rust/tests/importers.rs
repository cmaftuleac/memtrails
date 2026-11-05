//! The importers on invented sources: JSON Lines, a folder of notes, a Git history, a Slack export, a mailbox.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use base64::Engine;
use futures::{StreamExt, TryStreamExt};
use memtrails::importers::{
    batches, decode_encoded_words, from_eml, from_files, from_git, from_jsonl, from_mbox, from_slack_export,
    html_to_text, split_text, FilesOptions, GitOptions, GitTime, JsonlOptions, JsonlStats, MboxOptions, RecordStream,
    SlackOptions,
};
use memtrails::js::{date_parse, len16};
use memtrails::records::{At, MemoryRecord};
use serde_json::{json, Value};

async fn all(r: memtrails::Result<RecordStream>) -> Vec<MemoryRecord> {
    r.expect("the source opens").try_collect().await.expect("every record reads")
}

fn ms(at: &Option<At>) -> Option<f64> {
    match at {
        Some(At::Ms(x)) => Some(*x),
        _ => None,
    }
}

fn meta(r: &MemoryRecord, key: &str) -> Option<Value> {
    r.metadata.as_ref().and_then(|m| m.get(key).cloned())
}

// ---------------------------------------------------------------------------------------------------------------
// JSON Lines

#[tokio::test]
async fn jsonl_reads_record_shaped_lines_in_any_language_and_skips_and_counts_the_bad_ones() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("records.jsonl");
    let lines = [
        json!({ "id": "ja", "text": "会議は火曜日に延期されました 🎉", "title": "会議", "at": "2024-05-01T10:00:00Z" })
            .to_string(),
        "{ not json".into(),
        json!({ "text": "no id here" }).to_string(),
        String::new(),
        json!({ "id": 7, "text": "Întâlnirea de mâine e la București, în sala „Ștefan”." }).to_string(),
        json!({ "id": "ar", "text": "تم تأجيل الاجتماع إلى يوم الثلاثاء" }).to_string(),
        json!({ "id": "bad-time", "text": "x", "at": "whenever" }).to_string(),
        "[1, 2]".into(),
    ];
    fs::write(&path, format!("\u{feff}{}", lines.join("\r\n"))).unwrap();
    let skipped = Arc::new(Mutex::new(Vec::<String>::new()));
    let stats = Arc::new(Mutex::new(JsonlStats::default()));
    let seen = skipped.clone();
    let records = all(from_jsonl(
        &path,
        JsonlOptions {
            on_skip: Some(Arc::new(move |_line: &str, reason: &str, _n: usize| {
                seen.lock().unwrap().push(reason.into())
            })),
            stats: Some(stats.clone()),
            source: Some("notes".into()),
            ..Default::default()
        },
    )
    .await)
    .await;
    assert_eq!(records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["ja", "7", "ar"]);
    assert_eq!(records[0].title.as_deref(), Some("会議"));
    assert_eq!(records[0].text, "会議は火曜日に延期されました 🎉");
    assert_eq!(records[0].source.as_deref(), Some("notes"));
    assert_eq!(records[1].text, "Întâlnirea de mâine e la București, în sala „Ștefan”.");
    assert_eq!(*stats.lock().unwrap(), JsonlStats { lines: 7, records: 3, skipped: 4 });
    let skipped = skipped.lock().unwrap();
    assert_eq!(skipped.len(), 4);
    assert!(skipped[0].starts_with("not JSON"));
    assert!(skipped[1].contains("needs an id"));
}

#[tokio::test]
async fn jsonl_maps_any_object_shape_leaving_out_what_map_returns_none_for() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat.jsonl");
    let objs = [
        json!({ "msg_id": 1, "body": "Привет, как дела?", "who": "Олег", "ts": 1714557600000u64 }),
        json!({ "msg_id": 2, "body": "", "who": "bot" }),
        json!({ "msg_id": 3, "body": "नमस्ते, सब ठीक है।", "who": "प्रिया", "ts": 1714557660000u64 }),
    ];
    fs::write(&path, objs.iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n").unwrap();
    let reasons = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = reasons.clone();
    let records = all(from_jsonl(
        &path,
        JsonlOptions {
            map: Some(Arc::new(|o: &Value, _n: usize| {
                let body = o["body"].as_str().unwrap_or("");
                if body.is_empty() {
                    return Ok(None);
                }
                Ok(Some(MemoryRecord {
                    author: o["who"].as_str().map(Into::into),
                    at: o["ts"].as_f64().map(At::Ms),
                    kind: Some("message".into()),
                    ..MemoryRecord::new(o["msg_id"].to_string(), body)
                }))
            })),
            on_skip: Some(Arc::new(move |_l: &str, r: &str, _n: usize| seen.lock().unwrap().push(r.into()))),
            ..Default::default()
        },
    )
    .await)
    .await;
    let got: Vec<(String, String, String)> =
        records.iter().map(|r| (r.id.clone(), r.author.clone().unwrap_or_default(), r.text.clone())).collect();
    assert_eq!(
        got,
        [
            ("1".into(), "Олег".into(), "Привет, как дела?".into()),
            ("3".into(), "प्रिया".into(), "नमस्ते, सब ठीक है।".into())
        ]
    );
    assert_eq!(*reasons.lock().unwrap(), ["left out by map"]);
}

#[tokio::test]
async fn batches_group_items_as_they_come() {
    let got: Vec<Vec<i32>> = batches(futures::stream::iter(0..7), 3).collect().await;
    assert_eq!(got, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
}

// ---------------------------------------------------------------------------------------------------------------
// Files

fn paragraph(n: usize) -> String {
    format!("Paragraph {n}: {}— ação, naïve, 漢字.", "lorem ipsum dolor sit amet ".repeat(12))
}

fn big_text() -> String {
    format!("# Big\n\n{}", (0..30).map(paragraph).collect::<Vec<_>>().join("\n\n"))
}

fn notes_folder() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    fs::create_dir_all(d.join("notițe").join("日本")).unwrap();
    fs::create_dir_all(d.join(".git")).unwrap();
    fs::create_dir_all(d.join("node_modules").join("pkg")).unwrap();
    fs::write(
        d.join("notițe").join("plan.md"),
        [
            "---",
            "title: \"Planul pe trimestrul trei: priorități\"",
            "date: 2024-07-01",
            "author: Ioana Mărgărit",
            "tags: [buget, echipă]",
            "reviewers:",
            "  - Ana",
            "  - Mihai",
            "draft: true",
            "version: 3",
            "---",
            "# Not the title",
            "",
            "Bugetul pentru țară crește cu 5%.",
        ]
        .join("\n"),
    )
    .unwrap();
    fs::write(
        d.join("notițe").join("日本").join("メモ.md"),
        "Intro line\n\n# 東京オフィスの移転\n\n来月、新しいオフィスに移ります。🚚\n",
    )
    .unwrap();
    fs::write(d.join("readme.txt"), "Ελληνικά: καλημέρα κόσμε.\n").unwrap();
    let when =
        std::time::UNIX_EPOCH + std::time::Duration::from_millis(date_parse("2023-02-03T04:05:06Z").unwrap() as u64);
    fs::File::options().write(true).open(d.join("readme.txt")).unwrap().set_modified(when).unwrap();
    fs::write(
        d.join("page.html"),
        "<!doctype html><html><head><title>Caf&eacute; &amp; Cr&egrave;me</title>\n    <style>body { color: red }</style><script>var secret = \"<p>not text</p>\";</script></head>\n    <body><h1>Menu</h1><p>Prix&nbsp;: 3&nbsp;&euro; &mdash; <b>très</b>\n    bon</p><ul><li>Thé</li><li>&#x1F375; matcha &#8211; 抹茶</li></ul><!-- hidden --><p>Ünïcödé &lt;ok&gt;</p></body></html>",
    )
    .unwrap();
    fs::write(d.join(".git").join("config.md"), "# hidden").unwrap();
    fs::write(d.join(".secret.md"), "# hidden").unwrap();
    fs::write(d.join("node_modules").join("pkg").join("README.md"), "# dependency").unwrap();
    fs::write(d.join("image.png"), [0x89, 0x50, 0x4e, 0x47, 0, 0, 0]).unwrap();
    fs::write(d.join("big.md"), big_text()).unwrap();
    dir
}

#[tokio::test]
async fn files_reads_text_files_skipping_dot_folders_node_modules_and_other_extensions() {
    let dir = notes_folder();
    let records = all(from_files(
        dir.path(),
        FilesOptions { collection: Some("docs".into()), max_chars: Some(2000), ..Default::default() },
    )
    .await)
    .await;
    let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).filter(|id| !id.starts_with("big.md")).collect();
    assert_eq!(ids, ["notițe/plan.md", "notițe/日本/メモ.md", "page.html", "readme.txt"]);
    assert!(records.iter().all(|r| r.collection.as_deref() == Some("docs") && r.kind.as_deref() == Some("note")));

    let plan = records.iter().find(|r| r.id == "notițe/plan.md").unwrap();
    assert_eq!(plan.title.as_deref(), Some("Planul pe trimestrul trei: priorități"));
    assert_eq!(plan.author.as_deref(), Some("Ioana Mărgărit"));
    assert_eq!(ms(&plan.at), date_parse("2024-07-01"));
    assert!(plan.text.starts_with("# Not the title"));
    assert_eq!(
        serde_json::to_value(plan.metadata.as_ref().unwrap()).unwrap(),
        json!({ "date": "2024-07-01", "author": "Ioana Mărgărit", "tags": "buget, echipă", "reviewers": "Ana, Mihai", "draft": true, "version": 3 })
    );
    assert_eq!(
        records.iter().find(|r| r.id == "notițe/日本/メモ.md").unwrap().title.as_deref(),
        Some("東京オフィスの移転")
    );
    let readme = records.iter().find(|r| r.id == "readme.txt").unwrap();
    assert_eq!(readme.title.as_deref(), Some("readme"));
    assert_eq!(ms(&readme.at), date_parse("2023-02-03T04:05:06Z"));
    assert!(readme.url.as_deref().unwrap().starts_with("file://"));

    let page = records.iter().find(|r| r.id == "page.html").unwrap();
    assert_eq!(page.title.as_deref(), Some("Café & Crème"));
    for gone in ["secret", "color", "hidden", "<p>", "Café"] {
        assert!(!page.text.contains(gone), "{gone} in {}", page.text);
    }
    assert!(page.text.contains("Prix : 3 € — très bon"), "{}", page.text);
    assert!(page.text.contains("- Thé\n- 🍵 matcha – 抹茶"), "{}", page.text);
    assert!(page.text.contains("Ünïcödé <ok>"));
}

#[tokio::test]
async fn files_cuts_a_large_file_into_parts_that_share_a_thread_and_keep_every_word() {
    let dir = notes_folder();
    let parts: Vec<MemoryRecord> =
        all(from_files(dir.path(), FilesOptions { max_chars: Some(2000), ..Default::default() }).await)
            .await
            .into_iter()
            .filter(|r| r.id.starts_with("big.md"))
            .collect();
    assert!(parts.len() > 3);
    assert_eq!(
        parts.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        (1..=parts.len()).map(|i| format!("big.md#{i}")).collect::<Vec<_>>()
    );
    assert!(parts.iter().all(|p| p.thread.as_deref() == Some("big.md") && len16(&p.text) <= 2000));
    assert_eq!(parts[0].title.as_deref(), Some(format!("Big (1/{})", parts.len()).as_str()));
    assert_eq!(meta(&parts[1], "part"), Some(json!(2)));
    assert_eq!(meta(&parts[1], "parts"), Some(json!(parts.len())));
    let words = |s: &str| s.split_whitespace().count();
    let joined = parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join("\n\n");
    assert_eq!(words(&joined), words(&big_text()));
}

#[tokio::test]
async fn split_text_cuts_text_without_spaces_at_a_sentence_end_and_never_inside_a_character() {
    let cjk = "今日は晴れです。".repeat(40) + &"😀".repeat(200);
    let parts = split_text(&cjk, 150);
    assert_eq!(parts.concat(), cjk);
    assert!(parts[0].ends_with('。'));
}

#[tokio::test]
async fn html_blockquotes_drop_when_asked() {
    assert_eq!(
        html_to_text("<p>Yes.</p><blockquote>Old <blockquote>older</blockquote> text</blockquote><p>Bye</p>", true),
        "Yes.\n\nBye"
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Git

fn git_available() -> bool {
    Command::new("git").arg("--version").stdout(Stdio::null()).status().is_ok_and(|s| s.success())
}

fn git_in(repo: &Path, args: &[&str], env: &[(&str, &str)], input: Option<&str>) {
    use std::io::Write;
    let mut cmd = Command::new("git");
    cmd.args(args)
        .current_dir(repo)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    if let Some(text) = input {
        child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    } else {
        drop(child.stdin.take());
    }
    assert!(child.wait().unwrap().success(), "git {args:?}");
}

fn commit(repo: &Path, name: &str, email: &str, date: &str, message: &str) {
    let user = format!("user.name={name}");
    let mail = format!("user.email={email}");
    // `git commit -F -` reads the message from standard input; `--cleanup=verbatim` keeps it exactly as written.
    git_in(
        repo,
        &["-c", &user, "-c", &mail, "-c", "commit.gpgsign=false", "commit", "-q", "--cleanup=verbatim", "-F", "-"],
        &[("GIT_AUTHOR_DATE", date), ("GIT_COMMITTER_DATE", date)],
        Some(message),
    );
}

fn history() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path();
    git_in(repo, &["init", "-q", "-b", "main"], &[], None);
    fs::create_dir(repo.join("notițe")).unwrap();
    fs::write(repo.join("notițe").join("日本語.md"), "こんにちは\n").unwrap();
    fs::write(repo.join("README.md"), "hello\n").unwrap();
    git_in(repo, &["add", "-A"], &[], None);
    commit(
        repo,
        "Ștefan Müller-Łukasiewicz 王小明",
        "stefan@example.org",
        "2024-03-01T10:00:00+02:00",
        "Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n",
    );
    fs::write(repo.join("README.md"), "hello again\n").unwrap();
    git_in(repo, &["add", "-A"], &[], None);
    commit(
        repo,
        "Ana",
        "ana@example.org",
        "2024-03-02T11:30:00Z",
        &[
            "Tricky message with %H and --- and \u{1e}\u{1f} in it",
            "",
            "\u{1e}0123456789abcdef0123456789abcdef",
            "commit 0000000000000000000000000000000000000000",
            "Author: Nobody <nobody@example.org>",
            "-less but \u{1f}still odd",
            "",
            "Files: not really",
        ]
        .join("\n"),
    );
    dir
}

#[tokio::test]
async fn git_reads_every_commit_newest_first_whatever_its_message_holds() {
    if !git_available() {
        return;
    }
    let dir = history();
    let repo = dir.path();
    let name = repo.file_name().unwrap().to_string_lossy().into_owned();
    let records = all(from_git(repo, GitOptions::default()).await).await;
    assert_eq!(records.len(), 2);
    let (second, first) = (&records[0], &records[1]);
    assert_eq!(second.title.as_deref(), Some("Tricky message with %H and --- and \u{1e}\u{1f} in it"));
    assert!(second.text.contains("commit 0000000000000000000000000000000000000000"));
    assert!(second.text.contains("\u{1e}0123456789abcdef"));
    assert!(second.text.ends_with("Files: README.md"));
    assert_eq!(second.author.as_deref(), Some("Ana"));
    // git writes a UTC author date as `Z` from 2.39 on and as `+00:00` before: the time is compared, not its spelling.
    assert!(matches!(&second.at, Some(At::Text(_))));
    assert_eq!(memtrails::records::time_of(second.at.as_ref()), date_parse("2024-03-02T11:30:00Z"));

    assert!(first.id.len() == 40 && first.id.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(meta(first, "sha"), Some(json!(first.id)));
    assert_eq!(meta(first, "email"), Some(json!("stefan@example.org")));
    assert_eq!(first.author.as_deref(), Some("Ștefan Müller-Łukasiewicz 王小明"));
    assert_eq!(first.title.as_deref(), Some("Adaugă notițele în japoneză 🗾"));
    assert_eq!(
        first.text,
        "Adaugă notițele în japoneză 🗾\n\nCorpul mesajului: café, naïve, 漢字.\n\nFiles: README.md, notițe/日本語.md"
    );
    let Some(At::Text(at)) = &first.at else { panic!("a date") };
    assert_eq!(date_parse(at), date_parse("2024-03-01T08:00:00Z"));
    assert_eq!(
        (first.kind.as_deref(), first.thread.as_deref(), first.source.as_deref()),
        (Some("commit"), Some(name.as_str()), Some(name.as_str()))
    );
}

#[tokio::test]
async fn git_takes_max_count_paths_since_thread_and_a_link() {
    if !git_available() {
        return;
    }
    let dir = history();
    let repo = dir.path();
    assert_eq!(all(from_git(repo, GitOptions { max_count: Some(1), ..Default::default() }).await).await.len(), 1);
    let only_notes = all(from_git(
        repo,
        GitOptions {
            paths: vec!["notițe".into()],
            thread: Some("history".into()),
            url: Some(Arc::new(|sha: &str| format!("https://example.org/c/{sha}"))),
            ..Default::default()
        },
    )
    .await)
    .await;
    assert_eq!(
        only_notes.iter().map(|r| r.title.clone().unwrap()).collect::<Vec<_>>(),
        ["Adaugă notițele în japoneză 🗾"]
    );
    assert_eq!(only_notes[0].thread.as_deref(), Some("history"));
    assert_eq!(only_notes[0].url, Some(format!("https://example.org/c/{}", only_notes[0].id)));
    let since = date_parse("2024-03-02T00:00:00Z").unwrap();
    assert_eq!(
        all(from_git(repo, GitOptions { since: Some(GitTime::Ms(since)), ..Default::default() }).await).await.len(),
        1
    );
    assert_eq!(
        all(from_git(repo, GitOptions { branch: Some("main".into()), max_files: Some(1), ..Default::default() }).await)
            .await
            .len(),
        2
    );
    let capped = all(from_git(repo, GitOptions { max_files: Some(1), ..Default::default() }).await).await;
    assert!(capped[1].text.ends_with("Files: README.md, … and 1 more"), "{}", capped[1].text);
}

#[tokio::test]
async fn git_stops_when_the_reader_stops_early_and_fails_loudly_on_a_bad_revision() {
    if !git_available() {
        return;
    }
    let dir = history();
    let mut it = from_git(dir.path(), GitOptions::default()).await.unwrap();
    let first = it.next().await.unwrap().unwrap();
    assert_eq!(first.id.len(), 40);
    drop(it);
    let bad: memtrails::Result<Vec<_>> =
        from_git(dir.path(), GitOptions { rev: Some("no-such-branch".into()), ..Default::default() })
            .await
            .unwrap()
            .try_collect()
            .await;
    assert!(bad.unwrap_err().to_string().contains("git log failed"));
    assert!(from_git(dir.path(), GitOptions { rev: Some("--all".into()), ..Default::default() }).await.is_err());
}

#[tokio::test]
async fn git_reads_an_empty_repository_as_nothing() {
    if !git_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    git_in(dir.path(), &["init", "-q"], &[], None);
    assert!(all(from_git(dir.path(), GitOptions::default()).await).await.is_empty());
}

// ---------------------------------------------------------------------------------------------------------------
// Slack

fn slack_export() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let put = |path: &str, v: Value| {
        let p = dir.path().join(path);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    };
    put(
        "users.json",
        json!([
            { "id": "U1", "name": "ana", "real_name": "Ana Popescu", "profile": { "real_name": "Ana Popescu", "display_name": "ana" } },
            { "id": "U2", "name": "tanaka", "profile": { "real_name": "", "display_name": "田中" } },
            { "id": "U3", "name": "omar" },
        ]),
    );
    put("channels.json", json!([{ "id": "C1", "name": "general" }, { "id": "C2", "name": "proiecte" }]));
    put(
        "general/2024-03-01.json",
        json!([
            { "type": "message", "subtype": "channel_join", "user": "U3", "text": "<@U3> has joined the channel", "ts": "1709290000.000100" },
            { "type": "message", "user": "U1", "text": "Planul: <@U2> preia <#C2|proiecte>, vezi <https://example.org/doc|documentul> și <https://example.org/raw>. 1 &lt; 2 &amp;&amp; ok",
              "ts": "1709290800.000200", "thread_ts": "1709290800.000200", "reply_count": 1 },
            { "type": "message", "user": "U2", "text": "了解しました <@U1|ana> 👍", "ts": "1709291000.000300", "thread_ts": "1709290800.000200" },
            { "type": "message", "user": "U3", "text": "", "ts": "1709291100.000400", "files": [{ "name": "تقرير.pdf", "title": "التقرير الشهري" }] },
            { "type": "message", "user": "U3", "text": "", "ts": "1709291150.000450" },
            { "type": "message", "subtype": "bot_message", "bot_id": "B1", "username": "deploy-bot", "text": "Deployed <!here>",
              "attachments": [{ "title": "Build 42", "title_link": "https://example.org/b/42" }], "ts": "1709291200.000500" },
        ]),
    );
    put(
        "general/2024-03-02.json",
        json!([
            { "type": "message", "user": "U1", "text": "Mulțumesc! <#C1>", "ts": "1709377200.000100" },
            { "type": "message", "subtype": "channel_topic", "user": "U1", "text": "set the topic", "ts": "1709377300.000200" },
        ]),
    );
    put(
        "proiecte/2024-03-01.json",
        json!([{ "type": "message", "user": "U3", "text": "مرحبا بالجميع", "ts": "1709290900.000100" }]),
    );
    dir
}

#[tokio::test]
async fn slack_reads_every_message_that_says_something_with_names_links_and_threads() {
    let dir = slack_export();
    let records = all(from_slack_export(
        dir.path(),
        SlackOptions { workspace_url: Some("https://example.slack.com/".into()), ..Default::default() },
    )
    .await)
    .await;
    assert_eq!(
        records.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        [
            "C1:1709290800.000200",
            "C1:1709291000.000300",
            "C1:1709291100.000400",
            "C1:1709291200.000500",
            "C1:1709377200.000100",
            "C2:1709290900.000100"
        ]
    );
    let (root, reply, file, bot, later, other) =
        (&records[0], &records[1], &records[2], &records[3], &records[4], &records[5]);
    assert_eq!(root.text, "Planul: @田中 preia #proiecte, vezi documentul (https://example.org/doc) și https://example.org/raw. 1 < 2 && ok");
    assert_eq!(root.author.as_deref(), Some("Ana Popescu"));
    assert_eq!(root.kind.as_deref(), Some("message"));
    assert_eq!(root.collection.as_deref(), Some("general"));
    assert_eq!(root.thread.as_deref(), Some("general:1709290800.000200"));
    assert_eq!(ms(&root.at), Some(1709290800000.0));
    assert_eq!(root.url.as_deref(), Some("https://example.slack.com/archives/C1/p1709290800000200"));
    assert_eq!(meta(root, "replies"), Some(json!(1)));
    assert_eq!(reply.text, "了解しました @Ana Popescu 👍");
    assert_eq!(reply.author.as_deref(), Some("田中"));
    assert_eq!(reply.thread.as_deref(), Some("general:1709290800.000200"));
    assert_eq!(file.text, "File: التقرير الشهري");
    assert_eq!(file.author.as_deref(), Some("omar"));
    assert_eq!(file.thread.as_deref(), Some("general:1709291100.000400"));
    assert_eq!(bot.text, "Deployed @here\nAttachment: Build 42 (https://example.org/b/42)");
    assert_eq!(bot.author.as_deref(), Some("deploy-bot"));
    assert_eq!(later.text, "Mulțumesc! #general");
    assert_eq!(other.text, "مرحبا بالجميع");
    assert_eq!(other.collection.as_deref(), Some("proiecte"));
    assert_eq!(other.source, records[0].source);
}

#[tokio::test]
async fn slack_reads_only_the_channels_asked_for_into_the_collection_given_without_bots_when_asked() {
    let dir = slack_export();
    let records = all(from_slack_export(
        dir.path(),
        SlackOptions {
            channels: Some(vec!["#general".into()]),
            collection: Some("team".into()),
            skip_bots: true,
            ..Default::default()
        },
    )
    .await)
    .await;
    assert_eq!(records.len(), 4);
    assert!(records.iter().all(|r| r.collection.as_deref() == Some("team") && r.id.starts_with("C1:")));
}

// ---------------------------------------------------------------------------------------------------------------
// Mail

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn wrap(s: &str) -> String {
    s.as_bytes().chunks(76).map(|c| String::from_utf8_lossy(c).into_owned()).collect::<Vec<_>>().join("\n")
}

/// Quoted-printable, as a mail program writes it: bytes past ASCII, `=` and a line's last space as =XX, soft breaks
/// under 76.
fn qp(text: &str) -> String {
    text.split('\n')
        .map(|line| {
            let mut out = String::new();
            let mut width = 0;
            for byte in line.bytes() {
                let piece =
                    if byte > 126 || byte == 61 { format!("={byte:02X}") } else { char::from(byte).to_string() };
                if width + piece.len() > 72 {
                    out.push_str("=\n");
                    width = 0;
                }
                width += piece.len();
                out.push_str(&piece);
            }
            if out.ends_with(' ') || out.ends_with('\t') {
                let last = out.pop().unwrap();
                out.push_str(&format!("={:02X}", last as u8));
            }
            out
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const PLAIN: &str = "Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏\n\n> Ce s-a hotărât?\n> Nimic încă.\n\n-- \nAndré, București";
const HTML: &str = "<html><body><p>Bună ziua,</p><p>Atașez procesul-verbal.</p></body></html>";

fn first_message() -> String {
    // A Japanese subject cut in the middle of a character across two encoded-words on a folded line.
    let subject_bytes = "会議の議事録 — 2024年".as_bytes();
    let subject = format!("=?UTF-8?B?{}?=\n =?UTF-8?B?{}?=", b64(&subject_bytes[..4]), b64(&subject_bytes[4..]));
    [
        "From andre@example.org Mon Jan  1 10:00:00 2024".to_string(),
        "Return-Path: <andre@example.org>".into(),
        "From: =?iso-8859-1?Q?Andr=E9_Lef=E8vre?= <andre@example.org>".into(),
        "To: \"Ioana Mărgărit\" <ioana@example.org>,".into(),
        "\tbob@example.org".into(),
        "Cc: =?utf-8?q?=C5=81ukasz?= <lukasz@example.org>".into(),
        format!("Subject: {subject}"),
        "Date: Mon, 1 Jan 2024 10:00:00 +0100 (CET)".into(),
        "Message-ID: <m1@example.org>".into(),
        "MIME-Version: 1.0".into(),
        "Content-Type: multipart/mixed; boundary=\"outer\"".into(),
        String::new(),
        "This is a multi-part message in MIME format.".into(),
        "--outer".into(),
        "Content-Type: multipart/alternative;".into(),
        " boundary=inner".into(),
        String::new(),
        "--inner".into(),
        "Content-Type: text/plain; charset=utf-8".into(),
        "Content-Transfer-Encoding: quoted-printable".into(),
        String::new(),
        qp(PLAIN),
        "--inner".into(),
        "Content-Type: text/html; charset=utf-8".into(),
        "Content-Transfer-Encoding: base64".into(),
        String::new(),
        wrap(&b64(HTML.as_bytes())),
        "--inner--".into(),
        "--outer".into(),
        "Content-Type: application/pdf".into(),
        "Content-Disposition: attachment; filename*=UTF-8''%E8%AD%B0%E4%BA%8B%E9%8C%B2.pdf".into(),
        "Content-Transfer-Encoding: base64".into(),
        String::new(),
        b64(b"%PDF-1.4 not really"),
        "--outer--".into(),
        String::new(),
    ]
    .join("\n")
}

const SECOND: &str = "From ioana@example.org Mon Jan  1 12:00:00 2024\nFrom: Ioana Mărgărit <ioana@example.org>\nTo: andre@example.org\nSubject: Re: =?UTF-8?B?5Lya6K2w44Gu6K2w5LqL6Yyy?=\nDate: Mon, 1 Jan 2024 12:00:00 +0100\nMessage-ID: <m2@example.org>\nIn-Reply-To: <m1@example.org>\nReferences: <m1@example.org>\nContent-Type: text/plain\nContent-Transfer-Encoding: 8bit\n\nMulțumesc, am citit.\n>From the start it was clear.\n>>From a quoted line, kept escaped once.\n\nAndré Lefèvre wrote:\n> Bună ziua,\n> Atașez procesul-verbal.\n";

fn third_message() -> Vec<u8> {
    // Windows-1251 Cyrillic in base64 HTML, no Message-ID, a quoted reply in a blockquote.
    let cyrillic: &[u8] = &[0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2, 0x2c, 0x20, 0xec, 0xe8, 0xf0]; // "Привет, мир"
    let body = [b"<p>".as_slice(), cyrillic, b"</p><blockquote>old text</blockquote>".as_slice()].concat();
    [
        "From sergey@example.org Tue Jan  2 09:00:00 2024".to_string(),
        "From: sergey@example.org (Сергей)".into(),
        format!("Subject: =?gb2312?B?{}?= =?koi8-r?Q?=F0=D2=C9=D7=C5=D4?=", b64(&[0xd6, 0xd0, 0xce, 0xc4])),
        "Date: Tue, 02 Jan 2024 09:00:00 GMT".into(),
        "Content-Type: text/html; charset=windows-1251".into(),
        "Content-Transfer-Encoding: base64".into(),
        String::new(),
        b64(&body),
        String::new(),
    ]
    .join("\n")
    .into_bytes()
}

fn mailbox() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Inbox.mbox");
    let mut bytes = format!("{}\n{}\n", first_message(), SECOND).into_bytes();
    bytes.extend(third_message());
    fs::write(&path, bytes).unwrap();
    (dir, path)
}

#[tokio::test]
async fn mbox_reads_a_thread_of_multipart_quoted_printable_base64_and_encoded_word_mail() {
    let (_dir, path) = mailbox();
    let records = all(from_mbox(&path, MboxOptions::default()).await).await;
    assert_eq!(records.len(), 3);
    let (a, b, c) = (&records[0], &records[1], &records[2]);
    assert_eq!(a.id, "m1@example.org");
    assert_eq!(a.title.as_deref(), Some("会議の議事録 — 2024年"));
    assert_eq!(a.author.as_deref(), Some("André Lefèvre"));
    assert_eq!(a.kind.as_deref(), Some("email"));
    assert_eq!(a.thread.as_deref(), Some("m1@example.org"));
    assert_eq!(a.source.as_deref(), Some("Inbox"));
    assert_eq!(ms(&a.at), date_parse("2024-01-01T09:00:00Z"));
    assert_eq!(
        a.text,
        "Bună ziua,\n\nAtașez procesul-verbal al ședinței — vă rog să-l citiți până mâine. Mulțumesc frumos pentru tot ajutorul! 🙏\n\nAttachments: 議事録.pdf"
    );
    assert_eq!(
        serde_json::to_value(a.metadata.as_ref().unwrap()).unwrap(),
        json!({ "from": "André Lefèvre <andre@example.org>", "to": "\"Ioana Mărgărit\" <ioana@example.org>, bob@example.org", "cc": "Łukasz <lukasz@example.org>" })
    );
    assert_eq!(b.id, "m2@example.org");
    assert_eq!(b.title.as_deref(), Some("Re: 会議の議事録"));
    assert_eq!(b.author.as_deref(), Some("Ioana Mărgărit"));
    assert_eq!(b.thread.as_deref(), Some("m1@example.org"));
    assert_eq!(b.text, "Mulțumesc, am citit.\nFrom the start it was clear.\n\nAndré Lefèvre wrote:");
    assert_eq!(meta(b, "inReplyTo"), Some(json!("m1@example.org")));
    assert_eq!(c.title.as_deref(), Some("中文Привет"));
    assert_eq!(c.text, "Привет, мир");
    assert_eq!(c.author.as_deref(), Some("Сергей"));
    assert!(c.id.len() == 32 && c.id.bytes().all(|x| x.is_ascii_hexdigit()));
    assert_eq!(c.thread.as_deref(), Some(c.id.as_str()));
}

#[tokio::test]
async fn mbox_keeps_quoted_lines_and_the_signature_when_asked() {
    let (_dir, path) = mailbox();
    let records = all(from_mbox(
        &path,
        MboxOptions { keep_quoted: true, keep_signature: true, collection: Some("work".into()), ..Default::default() },
    )
    .await)
    .await;
    let (a, b, c) = (&records[0], &records[1], &records[2]);
    assert!(a.text.contains("> Ce s-a hotărât?"));
    assert!(a.text.contains("-- \nAndré, București"));
    assert!(b.text.contains(">From a quoted line, kept escaped once."));
    assert_eq!(c.text, "Привет, мир\n\nold text");
    assert_eq!(a.collection.as_deref(), Some("work"));
}

#[tokio::test]
async fn encoded_words_q_underscores_words_joined_across_spaces_plain_text_between_kept() {
    assert_eq!(
        decode_encoded_words("=?utf-8?Q?Gr=C3=BC=C3=9Fe_aus?= =?utf-8?Q?_K=C3=B6ln?= und mehr"),
        "Grüße aus Köln und mehr"
    );
    assert_eq!(decode_encoded_words("=?unknown-x?Q?abc?="), "abc");
}

#[tokio::test]
async fn eml_reads_one_message_with_crlf_line_endings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("note.eml");
    let msg = first_message();
    fs::write(&path, msg.split('\n').skip(1).collect::<Vec<_>>().join("\r\n")).unwrap();
    let records = all(from_eml(&path, MboxOptions::default()).await).await;
    let r = &records[0];
    assert_eq!(r.id, "m1@example.org");
    assert_eq!(r.title.as_deref(), Some("会議の議事録 — 2024年"));
    assert_eq!(r.source.as_deref(), Some("note"));
    assert!(r.text.starts_with("Bună ziua,\n\nAtașez"));
}
