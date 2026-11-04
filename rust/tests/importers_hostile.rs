//! Importers read files someone else wrote: no input may make them take more than linear time (a run of unclosed
//! tags, a long line of spaces), and the linear scans keep what the regexes they replaced did.

use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use futures::TryStreamExt;
use memtrails::importers::{
    from_eml, from_files, from_slack_export, html_title, html_to_text, FilesOptions, MboxOptions, RecordStream,
    SlackOptions,
};
use memtrails::js::date_parse;
use memtrails::records::{At, MemoryRecord};
use serde_json::json;

const N: usize = 200_000;

async fn all(r: memtrails::Result<RecordStream>) -> Vec<MemoryRecord> {
    r.expect("the source opens").try_collect().await.expect("every record reads")
}

/// The reference's bound is one second; a debug build is several times slower than a release one.
async fn timed(label: &str, f: impl std::future::Future<Output = ()>) {
    let limit = if cfg!(debug_assertions) { 4000 } else { 1000 };
    let t = Instant::now();
    f.await;
    let ms = t.elapsed().as_millis();
    assert!(ms < limit, "{label}: {ms} ms");
}

#[tokio::test]
async fn html_stays_linear_on_runs_of_unclosed_tags() {
    for open in [
        "<script",
        "<style",
        "<iframe",
        "<head",
        "<pre",
        "<![CDATA[",
        "<div",
        "<a",
        "<br",
        "<li",
        "</td",
        "<!x",
        "<title>",
        "<h1>",
        "<blockquote",
        "<<",
    ] {
        let s = open.repeat(N.div_ceil(open.len()));
        timed(open, async {
            html_to_text(&s, true);
            html_title(&s);
        })
        .await;
    }
}

#[tokio::test]
async fn html_drops_hidden_elements_keeps_pre_as_written_runs_an_unclosed_script_to_the_end() {
    assert_eq!(html_to_text("<p>a</p><SCRIPT type=\"x\">b</script ><p>c</p>", false), "a\n\nc");
    assert_eq!(html_to_text("<p>a</p><iframe src=x>keep<p>b</p>", false), "a\nkeep\nb");
    assert_eq!(html_to_text("a <script>var x = 1; <p>never</p>", false), "a");
    assert_eq!(html_to_text("<head><title>t</title><script>s</script></head><p>x   y</p>", false), "x y");
    assert_eq!(html_to_text("<p>a   b</p><pre>  x\r\n   y</pre><p>c   d</p>", false), "a b\n\nx\ny\n\nc d");
    assert_eq!(html_to_text("<pre>  open   <p>x</p>", false), "open\nx");
    assert_eq!(html_to_text("<![CDATA[x < y]]> z", false), "x < y z");
    assert_eq!(html_to_text("a <![CDATA[ unclosed", false), "a <![CDATA[ unclosed");
    assert_eq!(html_to_text("<blockquote>q<blockquote>qq</blockquote></blockquote>reply", true), "reply");
    assert_eq!(html_to_text("<p>a</p> 1 < 2", false), "a\n1 < 2");
}

#[tokio::test]
async fn html_finds_the_title_else_the_first_h1_in_the_first_64_kb() {
    assert_eq!(
        html_title("<html><title lang=ro>Caf&eacute; <b>bun</b></title ><h1>H</h1>").as_deref(),
        Some("Café bun")
    );
    assert_eq!(html_title("<title> </title><h1 class=x>Head<br>line</h1>").as_deref(), Some("Head line"));
    assert_eq!(html_title("<title>open"), None);
    assert_eq!(html_title(&format!("{}<title>late</title>", " ".repeat(70_000))), None);
}

#[tokio::test]
async fn files_read_in_linear_time() {
    let sp = " ".repeat(N);
    let files: Vec<(&str, String)> = vec![
        ("atx.md", format!("# a{sp}b")),
        ("atx-close.md", format!("# a{sp}#")),
        ("setext.md", format!("a{sp}b\n===")),
        ("eq.md", format!("{}x\n{}", "=".repeat(N), "=".repeat(N))),
        ("front.md", format!("---\ntitle: a{sp}b\nk: v{sp}x\n---\nbody")),
        ("front.toml.md", format!("+++\na{sp}\nb = c{sp}d\n+++\n")),
        ("title.rst", format!("a{sp}b\n{}x", "=".repeat(N))),
        ("stars.org", format!("* {}", "*".repeat(N))),
        ("eq.adoc", "=".repeat(N)),
        ("tags.html", "<title>".repeat(N / 7)),
        ("hashes.md", format!("{}\n# {}", "#".repeat(N), "# ".repeat(N / 2))),
    ];
    for (name, text) in files {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(name), text).unwrap();
        timed(name, async {
            all(from_files(dir.path(), FilesOptions::default()).await).await;
        })
        .await;
    }
}

#[tokio::test]
async fn files_keep_the_titles_the_regexes_found() {
    let dir = tempfile::tempdir().unwrap();
    let cases: [(&str, &str, &str); 11] = [
        ("atx.md", "intro\n# Plan   ##  \nbody", "Plan"),
        ("atx-hash.md", "# C# tips", "C# tips"),
        ("atx-only.md", "# ###", "###"),
        ("setext.md", "Planul   \n=====\nbody", "Planul"),
        ("over.rst", "=====\n Titlu \n=====\n\ntext", "Titlu"),
        ("under.rst", "Titlu\n~~~~~\n", "Titlu"),
        ("short.rst", "Titlu\n==\nNot\n---\n", "Not"),
        ("doc.adoc", "== Sub\n= Document  \n", "Document"),
        ("notes.org", "#+TITLE:  Notițe\n* Head", "Notițe"),
        ("head.org", "text\n** Capitol 2\n", "Capitol 2"),
        ("cr.md", "intro\r# Old Mac\rbody", "Old Mac"),
    ];
    for (name, text, _) in &cases {
        fs::write(dir.path().join(name), text).unwrap();
    }
    let titles: HashMap<String, String> = all(from_files(dir.path(), FilesOptions::default()).await)
        .await
        .into_iter()
        .map(|r| (r.id, r.title.unwrap_or_default()))
        .collect();
    for (name, _, title) in cases {
        assert_eq!(titles.get(name).map(String::as_str), Some(title), "{name}");
    }
}

#[tokio::test]
async fn files_read_front_matter_as_before() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.md"), "---\ntitle: T # a comment\nurl: http://x.org/a:b\nk:v\ncount: 3\n---\nbody")
        .unwrap();
    fs::write(dir.path().join("b.md"), "+++\ntitle = \"TOML\"\nn = 2 # c\n+++\nbody").unwrap();
    let records = all(from_files(dir.path(), FilesOptions::default()).await).await;
    assert_eq!(records[0].title.as_deref(), Some("T"));
    assert_eq!(
        serde_json::to_value(records[0].metadata.as_ref().unwrap()).unwrap(),
        json!({ "url": "http://x.org/a:b", "count": 3 })
    );
    assert_eq!(records[1].title.as_deref(), Some("TOML"));
    assert_eq!(serde_json::to_value(records[1].metadata.as_ref().unwrap()).unwrap(), json!({ "n": 2 }));
}

#[tokio::test]
async fn email_reads_in_linear_time() {
    let sp = " ".repeat(N);
    let emls = [
        ("from-at.eml", format!("From: {}(x\nSubject: s\n\nhi\n", "a@".repeat(N / 2))),
        ("from-spaces.eml", format!("From: a{sp}<b\nSubject: s\n\nhi\n")),
        ("qp.eml", format!("From: a@b.c\nContent-Transfer-Encoding: quoted-printable\n\na{sp}b\n")),
        ("date.eml", format!("From: a@b.c\nDate: {}\nSubject: s\n\nhi\n", "(".repeat(N))),
    ];
    for (name, text) in emls {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        fs::write(&path, text).unwrap();
        timed(name, async {
            all(from_eml(&path, MboxOptions::default()).await).await;
        })
        .await;
    }
}

#[tokio::test]
async fn email_still_names_the_sender_and_reads_the_date() {
    let dir = tempfile::tempdir().unwrap();
    let mut authors = Vec::new();
    for (i, from) in
        ["\"Pop, Ana\" <ana@x.org>", "<ana@x.org>", "ana@x.org (Ana Pop)", "ana@x.org ()", "Ana"].iter().enumerate()
    {
        let path = dir.path().join(format!("{i}.eml"));
        fs::write(&path, format!("From: {from}\nDate: Fri, 1 Mar 2024 10:00:00 +0000 (UTC)\nSubject: s\n\nhi\n"))
            .unwrap();
        let r = all(from_eml(&path, MboxOptions::default()).await).await.remove(0);
        authors.push(r.author.unwrap_or_default());
        assert_eq!(r.at, date_parse("2024-03-01T10:00:00Z").map(At::Ms));
    }
    assert_eq!(authors, ["Pop, Ana", "ana@x.org", "Ana Pop", "ana@x.org ()", "Ana"]);
}

#[tokio::test]
async fn a_slack_export_naming_folders_outside_itself_reads_none_of_them() {
    let base = tempfile::tempdir().unwrap();
    let root = base.path().join("export");
    let outside = base.path().join("secret");
    fs::create_dir_all(root.join("general")).unwrap();
    fs::create_dir(&outside).unwrap();
    let msg = |text: &str, ts: &str| json!([{ "type": "message", "user": "U1", "text": text, "ts": ts }]).to_string();
    fs::write(outside.join("2024-01-01.json"), msg("outside", "1704100000.0001")).unwrap();
    fs::write(root.join("general").join("2024-01-01.json"), msg("inside", "1704100000.0002")).unwrap();
    fs::write(
        root.join("channels.json"),
        json!([
            { "id": "C1", "name": "general" }, { "id": "C2", "name": "../secret" }, { "id": "C3", "name": ".." },
            { "id": "C4", "name": outside.to_string_lossy() }, { "id": "C5", "name": "a/../../secret" }, { "id": "C6", "name": "..\\secret" },
        ])
        .to_string(),
    )
    .unwrap();
    fs::write(root.join("dms.json"), json!([{ "id": "../secret", "members": ["U1"] }]).to_string()).unwrap();
    let texts: Vec<String> =
        all(from_slack_export(&root, SlackOptions::default()).await).await.into_iter().map(|r| r.text).collect();
    assert_eq!(texts, ["inside"]);
}
