//! `Date.parse` as V8 reads the forms people and mail programs write, in UTC — the Python port's answer key
//! (`conformance/fixtures/python-dates.json`, made by the reference with TZ=UTC), read here too.

#[test]
fn date_parse_reads_what_v8_reads() {
    // The only test in this binary: the time zone is this process's own.
    std::env::set_var("TZ", "UTC");
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance/fixtures/python-dates.json");
    let f: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut wrong = Vec::new();
    for c in f["cases"].as_array().unwrap() {
        let s = c["s"].as_str().unwrap();
        let got = memtrails::js::date_parse(s);
        if got != c["ms"].as_f64() {
            wrong.push(format!("{s:?}: got {got:?}, want {:?}", c["ms"].as_f64()));
        }
    }
    // Hostile numbers, as V8 reads them: leading zeros skipped, nine significant digits at most, an offset's seconds
    // in 32 bits — never a panic, never a date where V8 has none (answers from node, TZ=UTC).
    let hostile: serde_json::Value = serde_json::from_str(
        r#"[["Jan 1 99999999999999999999",null],["Jan 1 2024 10:00 +99999999999999999:00",886700848000],["Jan 1 2024 10:00 -99999999999999999:00",2521505552000],["Jan 1 2024 10:00 +1234567890123:00",-358705712000],["Jan 1 2024 10:00 +596523:00",-443379600000],["Jan 1 2024 10:00 +596524:00",null],["Jan 1 2024 10:00 +12345",null],["Jan 1 2024 10:00 +123",1704098220000],["Jan 1 2024 10:00:00.99999999999999999999",1704103200999],["Jan 1 2024 10:00:00.0000000000000000001",1704103200000],["Jan 99999999999999999999 2024",null],["2024/1/99999999999999999999",null],["Jan 1 2024 99999999999999999999:00",null],["Jan 1 1000000000",null],["Jan 1 2024 10:00 GMT+99999999999999999999",null],["Jan 1 2024 10:00 +00000000000000000001:00",1704099600000],["1/1/0000000000000002024",1704067200000]]"#,
    )
    .unwrap();
    for c in hostile.as_array().unwrap() {
        let s = c[0].as_str().unwrap();
        let got = memtrails::js::date_parse(s);
        if got != c[1].as_f64() {
            wrong.push(format!("{s:?}: got {got:?}, want {:?}", c[1].as_f64()));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
