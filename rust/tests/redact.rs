//! The secret filter: what it must take out, what ordinary prose it must leave, and that no hostile string makes it
//! slow (it runs on every prompt and every model answer kept).

use std::time::Instant;

use memtrails::redact::redact_string;

fn gone(input: &str, secret: &str) {
    let out = redact_string(input);
    assert!(!out.contains(secret), "{input} → {out}");
    assert!(out.contains("<redacted"), "{input} → {out}");
}

/// Fake keys put together here, so a secret scanner reading this file sees no key.
fn fake(parts: &[&str]) -> String {
    parts.concat()
}

#[test]
fn takes_the_value_of_a_secret_named_variable_whatever_its_prefix() {
    gone("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI");
    gone("export GITHUB_TOKEN=abcdef1234567890abcdef", "abcdef1234567890");
    gone("DB_PASSWORD=hunter2hunter", "hunter2");
    gone("PGPASSWORD=s3cretpw psql -h db", "s3cretpw");
    gone("access_token: abc123def456ghi", "abc123def456");
    gone("client_secret=zzzyyyxxx111", "zzzyyyxxx111");
    gone("AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net", "abc+def");
    gone("Endpoint=sb://x.servicebus.windows.net/;SharedAccessKey=Zm9vYmFyYmF6", "Zm9vYmFyYmF6");
    gone("password = hunter2", "hunter2");
    assert!(redact_string("AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net")
        .contains(";EndpointSuffix=core.windows.net"));
}

#[test]
fn takes_json_style_secrets_spaces_and_escaped_quotes_included() {
    assert_eq!(redact_string(r#"{"api_key": "abcd1234efgh5678"}"#), r#"{"api_key": "<redacted>"}"#);
    assert_eq!(redact_string(r#"{"password":"hunter2 with space"}"#), r#"{"password":"<redacted>"}"#);
    gone(r"{'client_secret': 'q\'uote secret'}", "uote");
    gone(r#"{"refresh_token":"1//0gAbCdEf"}"#, "0gAbCdEf");
}

#[test]
fn takes_provider_key_formats() {
    gone(&fake(&["ci uses glpat", "-abcdefghij1234567890 for pushes"]), "glpat-");
    gone(&fake(&["HF hf", "_abcdefghijklmnopqrstuvwxyzABCD"]), "hf_abc");
    gone(&fake(&["npm", "_abcdefghijklmnopqrstuvwxyz0123456789"]), "npm_abc");
    gone("temporary ASIAABCDEFGHIJKLMNOP", "ASIAABCDEFGHIJKLMNOP");
    gone("AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP");
    gone(
        &fake(&["SG", ".abcdefghijklmnopqrstuv", ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"]),
        "abcdefghijklmnopqrstuv",
    );
    gone(&fake(&["bot 123456789", ":ABCdefGHIjklMNOpqrSTUvwxYZ012345678 ok"]), "ABCdefGHI");
    gone("sk-abcdefghijklmnop1234", "sk-abc");
    gone("ghp_abcdefghijklmnopqrstuvwxyz", "ghp_");
}

#[test]
fn takes_the_credential_of_an_authorization_header_scheme_or_not() {
    assert_eq!(redact_string("Authorization: token ghx1234567890abcdef"), "Authorization: token <redacted>");
    gone(r#"-H "Authorization: Bearer abc.def.ghijklmnop""#, "abc.def");
    gone("authorization=abcdefghijkl", "abcdefghijkl");
}

#[test]
fn takes_the_whole_password_from_a_url() {
    assert_eq!(redact_string("postgres://user:p@ss@host/db"), "postgres://user:<redacted>@host/db");
    assert_eq!(redact_string("https://user:pa/ss@host/x"), "https://user:<redacted>@host/x");
    assert_eq!(redact_string("see https://u:pw@example.org and ok"), "see https://u:<redacted>@example.org and ok");
    assert_eq!(redact_string("redis://:s3cret@cache:6379"), "redis://:<redacted>@cache:6379");
}

#[test]
fn still_takes_what_it_took_before() {
    gone("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----", "MIIabc");
    gone("-----BEGIN PRIVATE KEY-----\nMIIunterminated", "MIIunterminated");
    assert_eq!(redact_string("mail ana.pop@example.org now"), "mail <redacted:email>@example.org now");
    gone("https://x.org/cb?access_token=abcdef&x=1", "abcdef");
    gone("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "eyJhbGci");
    assert_eq!(redact_string("/Users/ana/projects/x"), "~/projects/x");
    assert_eq!(redact_string(&"a".repeat(60)), "<redacted:entropy>");
}

#[test]
fn leaves_ordinary_prose_counts_and_commit_shas_alone() {
    for s in [
        "the password field is required",
        "token-based auth is simpler",
        "max_tokens: 4096",
        "token_count: 12",
        "fixed in 0123456789abcdef0123456789abcdef01234567 yesterday",
        "commit 9fceb02d0ae598e95dc970b74767f19372d61af8",
        "author: Jane Doe",
        "key: value",
        "a secret santa party",
        "http://localhost:8080/path",
        "meet at 10:30, then 12:45",
    ] {
        assert_eq!(redact_string(s), s);
    }
}

#[test]
fn takes_40_hex_characters_when_a_secret_name_says_what_they_are() {
    gone("GITHUB_TOKEN=0123456789abcdef0123456789abcdef01234567", "0123456789abcdef");
}

#[test]
fn stays_linear_on_hostile_input() {
    let n = 200_000;
    let hostile = [
        "a.".repeat(n / 2) + "@",
        "x@".to_string() + &"a.".repeat(n / 2),
        "-a".repeat(n / 2) + "@",
        "a@".repeat(n / 2),
        "eyJ-".repeat(n / 4),
        "eyJaaaaaaaaa.".repeat(n / 13),
        "a://b:".repeat(n / 6),
        "a://b:@".repeat(n / 7),
        "x://u:p@".to_string() + &"a".repeat(n),
        "-----BEGIN PRIVATE KEY-----".repeat(n / 27),
        "password".to_string() + &" ".repeat(n),
        "a_".repeat(n / 2) + "=",
        "password=".repeat(n / 9),
        "\"password\": \"".repeat(n / 14),
        "Bearer ".repeat(n / 7),
        "1234567890:".repeat(n / 11),
        "SG.".to_string() + &"a".repeat(n),
        "a:".repeat(n / 2),
        "/Users/".repeat(n / 7),
        " ".repeat(n),
    ];
    // A debug build is several times slower than a release one: the bound is the reference's second, times four.
    let limit_ms = if cfg!(debug_assertions) { 4000 } else { 1000 };
    for s in &hostile {
        let t = Instant::now();
        redact_string(s);
        let ms = t.elapsed().as_millis();
        if std::env::var_os("SHOW").is_some() {
            println!("{} {ms}", &s[..12.min(s.len())]);
        }
        assert!(ms < limit_ms, "{}: {ms} ms", &s[..20.min(s.len())]);
    }
}
