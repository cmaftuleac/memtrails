//! Records as memory keeps them: titles, bodies, authors, fingerprints, conversations cut at each request.

use memtrails::episodes::{episode_body, render_for_extraction};
use memtrails::records::{
    collection_key, conversation_entries, entry_of, invalid_record, record_uid, time_of, At, Conversation,
    MemoryRecord, Role, Turn,
};

fn now() -> f64 {
    memtrails::js::date_parse("2026-05-01T00:00:00Z").unwrap()
}

#[test]
fn keeps_a_titled_record_as_title_and_text_with_its_author_kind_collection_thread_and_reference() {
    let mut meta = indexmap::IndexMap::new();
    meta.insert("to".to_string(), serde_json::json!("ops"));
    let r = MemoryRecord {
        id: "m1".into(),
        source: Some("mailbox".into()),
        kind: Some("email".into()),
        title: Some("Proxy certificate".into()),
        author: Some("Ana".into()),
        text: "It expired on the 1st.\nThe renewal job had been removed.".into(),
        collection: Some("Ops team".into()),
        thread: Some("t-9".into()),
        at: Some(At::from("2026-04-02")),
        url: Some("mid:1".into()),
        metadata: Some(meta),
    };
    let e = entry_of(&r, now(), 0);
    assert_eq!(e.engine, "email");
    assert_eq!(e.session, "t-9");
    assert_eq!(e.project, "Ops-team");
    assert_eq!(e.request, "Proxy certificate");
    assert_eq!(e.reply, "Ana: It expired on the 1st.\nThe renewal job had been removed.");
    assert_eq!(e.ts, memtrails::js::date_parse("2026-04-02").unwrap());
    assert_eq!(e.files, ["mid:1"]);
    let rec = e.record.clone().unwrap();
    assert_eq!((rec.id.as_str(), rec.source.as_deref(), rec.author.as_deref()), ("m1", Some("mailbox"), Some("Ana")));
    assert_eq!(e.uid, record_uid("m1", Some("mailbox")));
    assert_eq!(episode_body(&e), "Proxy certificate\nAna: It expired on the 1st.\nThe renewal job had been removed.");
    assert_eq!(
        render_for_extraction(&e, 3),
        "## Record 3 · 2026-04-02 00:00 · email\nProxy certificate\nAna: It expired on the 1st.\nThe renewal job had been removed."
    );
}

#[test]
fn takes_the_first_line_as_the_title_of_an_untitled_record_in_any_script() {
    let e = entry_of(&MemoryRecord::new("n", "会议记录：部署推迟到周五\n原因是证书过期。"), now(), 0);
    assert_eq!(e.title, "会议记录：部署推迟到周五");
    assert_eq!(e.reply, "原因是证书过期。");
    assert_eq!(e.ts, now());
    assert_eq!(e.engine, "record");
    assert_eq!(e.session, e.uid);
    assert_eq!(collection_key(None), "default");
}

#[test]
fn changes_its_fingerprint_when_anything_about_it_changes_and_only_then() {
    let r = MemoryRecord { author: Some("Ion".into()), ..MemoryRecord::new("x", "a\nb") };
    assert_eq!(entry_of(&r, now(), 0).fingerprint, entry_of(&r, now() + 1.0, 0).fingerprint);
    let other = MemoryRecord { text: "a\nc".into(), ..r.clone() };
    assert_ne!(entry_of(&other, now(), 0).fingerprint, entry_of(&r, now(), 0).fingerprint);
    let mut meta = indexmap::IndexMap::new();
    meta.insert("k".to_string(), serde_json::json!(1));
    let with_meta = MemoryRecord { metadata: Some(meta), ..r.clone() };
    assert_ne!(entry_of(&with_meta, now(), 0).fingerprint, entry_of(&r, now(), 0).fingerprint);
}

#[test]
fn says_what_is_wrong_with_a_record() {
    assert!(invalid_record(&MemoryRecord::new("", "x")).unwrap().contains("id"));
    assert!(invalid_record(&MemoryRecord::new("a", "  ")).unwrap().contains("no text"));
    let bad_time = MemoryRecord { at: Some(At::from("not a date")), ..MemoryRecord::new("a", "x") };
    assert!(invalid_record(&bad_time).unwrap().contains("time"));
    let titled = MemoryRecord { title: Some("only a title".into()), ..MemoryRecord::new("a", "") };
    assert!(invalid_record(&titled).is_none());
    assert_eq!(time_of(Some(&At::Ms(5.0))), Some(5.0));
    assert_eq!(time_of(Some(&At::from(""))), None);
}

#[test]
fn cuts_a_conversation_at_each_time_the_person_speaks() {
    let turn = |role, text: &str, author: Option<&str>| Turn {
        role,
        text: text.into(),
        author: author.map(Into::into),
        at: None,
    };
    let c = Conversation {
        id: "c1".into(),
        collection: Some("p".into()),
        source: None,
        at: Some(At::Ms(now())),
        turns: vec![
            turn(Role::Assistant, "Hello, how can I help?", None),
            turn(Role::User, "Deploy failed.", None),
            turn(Role::User, "Logs say TLS.", None),
            turn(Role::Assistant, "The certificate expired.", None),
            turn(Role::Assistant, "Renew it with the cron job.", None),
            turn(Role::User, "Thanks", Some("Ana")),
        ],
    };
    let es = conversation_entries(&c, now());
    let pairs: Vec<(String, String)> = es.iter().map(|e| (e.request.clone(), e.reply.clone())).collect();
    assert_eq!(
        pairs,
        [
            ("Hello, how can I help?".into(), "Hello, how can I help?".into()),
            ("Deploy failed.\nLogs say TLS.".into(), "The certificate expired.\nRenew it with the cron job.".into()),
            ("Ana: Thanks".into(), String::new()),
        ]
    );
    assert!(es.iter().all(|e| e.engine == "conversation" && e.session == "c1"));
    assert_eq!(es.iter().map(|e| e.turn).collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(es[1].said, es[1].reply);
    assert!(render_for_extraction(&es[1], 1).contains("Agent said:\nThe certificate expired."));
}
