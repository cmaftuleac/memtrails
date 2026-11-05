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

#[test]
fn metadata_with_number_keys_is_fingerprinted_in_the_order_javascript_gives_its_keys() {
    // JavaScript puts integer-like keys first, ascending, whatever order they came in (answers from node).
    let v: serde_json::Value = serde_json::from_str(
        r#"{"b":1,"10":2,"2":3,"a":{"z":1,"1":2},"01":5,"4294967295":6,"4294967294":7,"-1":8,"1.5":9,"0":10}"#,
    )
    .unwrap();
    assert_eq!(
        memtrails::js::stringify(&v),
        r#"{"0":10,"2":3,"10":2,"4294967294":7,"b":1,"a":{"1":2,"z":1},"01":5,"4294967295":6,"-1":8,"1.5":9}"#
    );
    let meta: indexmap::IndexMap<String, serde_json::Value> =
        serde_json::from_str(r#"{"b":1,"10":"x","2":true}"#).unwrap();
    let r = MemoryRecord { title: Some("T".into()), metadata: Some(meta), ..MemoryRecord::new("m1", "body") };
    let e = entry_of(&r, now(), 0);
    assert_eq!(e.fingerprint, "b297e75415e136ae7b19ac32d906f3216972a6f2e978e726616fbf60c4ffab55");
    // And kept in that order, as the reference's log line has it.
    let kept: Vec<String> = e.record.unwrap().metadata.unwrap().keys().cloned().collect();
    assert_eq!(kept, ["2", "10", "b"]);
}

#[test]
fn a_date_past_year_9999_is_cut_from_its_iso_text_as_javascript_cuts_it() {
    // `toISOString().slice(0, n)` on a six-digit year (answers from node): the cut is by place, not by the tail.
    let t = 253_407_762_367_000.0; // +010000-03-04T05:06:07.000Z
    let cal = memtrails::episodes::calendar_of(&[t]);
    assert!(cal.starts_with("- Sat +010000-03: yesterday "), "{cal}");
    assert!(cal.contains("last month +010000 ·"), "{cal}");
    let e = entry_of(&MemoryRecord { at: Some(At::Ms(t)), ..MemoryRecord::new("m1", "text") }, now(), 0);
    assert!(
        render_for_extraction(&e, 1).starts_with("## Record 1 · +010000-03-04 05 · record"),
        "{}",
        render_for_extraction(&e, 1)
    );
}

#[test]
fn a_turn_said_by_the_system_loads_and_is_not_the_persons() {
    // The reference reads any role but "user" as the other side.
    let c: Conversation = serde_json::from_value(serde_json::json!({
        "id": "c2",
        "turns": [
            { "role": "system", "text": "You are a helpful assistant." },
            { "role": "user", "text": "Deploy failed." },
            { "role": "tool", "text": "exit 1" },
        ]
    }))
    .unwrap();
    let es = conversation_entries(&c, now());
    let pairs: Vec<(String, String)> = es.iter().map(|e| (e.request.clone(), e.reply.clone())).collect();
    // As an assistant that opened: the first record its own, then the person's request with what answered it.
    let want = [("You are a helpful assistant.", "You are a helpful assistant."), ("Deploy failed.", "exit 1")];
    assert_eq!(pairs, want.map(|(a, b)| (a.to_string(), b.to_string())));
}
