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
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
