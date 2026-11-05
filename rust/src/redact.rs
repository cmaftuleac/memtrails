//! Secrets are scrubbed from what memory sends to a language model and from what consolidation keeps: provider keys,
//! tokens, credentials in URLs, private keys, e-mail local parts, JWTs, `key=value` secrets; home directories become
//! `~`; a long high-entropy run last. The records you add are kept as you wrote them.
//!
//! The input can be anything a record held, so every rule takes linear time on a hostile string (a long run of what
//! almost matches): the `regex` crate's automata guarantee it, and the three rules the reference writes with a
//! look-ahead (a Telegram token, a URL's password, a JWT) are scans written out here, each bounded as its pattern is.

use std::sync::LazyLock;

use regex::{Captures, Regex, RegexBuilder};

use crate::js;

fn build(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .size_limit(64 << 20)
        .dfa_size_limit(64 << 20)
        .build()
        .unwrap_or_else(|e| panic!("memtrails: invalid built-in pattern {pattern}: {e}"))
}

/// JavaScript's `\b` without the `u` flag: ASCII word characters only.
const WB: &str = r"(?-u:\b)";

/// A name that says its value is a secret: `password`, `GITHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY`, `PGPASSWORD`,
/// `client_secret`, `AccountKey`. The secret word must end the name (`token_count`, `max_tokens` are not secrets), and
/// any prefix is allowed (`_` is a word character, so `\btoken` alone misses `GITHUB_TOKEN`). Letters match either
/// case, ASCII only, as the reference's `i` flag does.
const SECRET_NAME: &str = r"[A-Za-z0-9_.-]{0,64}?(?i-u:api[_-]?key|apikey|(?:access|secret|private|account|signing|encryption|master|client)[_-]?key|token|secret|password|passwd|passphrase|authorization)";

/// `name=value`, `name: value`, `"name": "value"`, `Authorization: token X` (the scheme word is kept, its credential
/// not).
static KEY_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    build(&format!(
        r#"{WB}({SECRET_NAME})(["']?[ \t]{{0,16}}[=:][ \t]{{0,16}})(?:(["'])((?:[^"'\\\n]|\\[^\n\r\u{{2028}}\u{{2029}}]){{1,512}})|((?:(?i-u:Bearer|Basic|Digest|token|Bot)[ \t]{{1,16}})?)([^{s}&"';]+))"#,
        s = js::SPACE_CLASS
    ))
});

static SECRET_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let s = js::SPACE_CLASS;
    vec![
        (build(&format!(r"{WB}sk-[A-Za-z0-9_-]{{16,}}")), "<redacted:sk>"),
        (build(&format!(r"{WB}sk_(?:live|test)_[A-Za-z0-9]{{16,}}")), "<redacted:stripe>"),
        (build(&format!(r"{WB}pk_(?:live|test)_[A-Za-z0-9]{{16,}}")), "<redacted:stripe-pub>"),
        (build(&format!(r"{WB}github_pat_[A-Za-z0-9_]{{20,}}")), "<redacted:github>"),
        (build(&format!(r"{WB}gh[porsu]_[A-Za-z0-9]{{20,}}")), "<redacted:github>"),
        (build(&format!(r"{WB}glpat-[A-Za-z0-9_-]{{20,}}")), "<redacted:gitlab>"),
        (build(&format!(r"{WB}hf_[A-Za-z0-9]{{30,}}")), "<redacted:huggingface>"),
        (build(&format!(r"{WB}npm_[A-Za-z0-9]{{36,}}")), "<redacted:npm>"),
        (build(&format!(r"{WB}xox[abprs]-[A-Za-z0-9-]{{10,}}")), "<redacted:slack>"),
        (build(&format!(r"{WB}(?:AKIA|ASIA)[0-9A-Z]{{16}}{WB}")), "<redacted:aws>"),
        (build(&format!(r"{WB}AIza[A-Za-z0-9_-]{{20,}}")), "<redacted:google>"),
        (build(&format!(r"{WB}SG\.[A-Za-z0-9_-]{{16,}}\.[A-Za-z0-9_-]{{16,}}")), "<redacted:sendgrid>"),
        // Telegram, URL credentials: scans below (`telegram`, `url_credentials`), in this place of the order.
        (build(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+"), "https://hooks.slack.com/services/<redacted>"),
        (
            build(r"https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+"),
            "<redacted:discord-webhook>",
        ),
        // Local part and domain bounded as RFC 5321 bounds them: unbounded, a long run of `a.a.a…` without an `@` is
        // quadratic for a backtracking engine.
        (
            build(&format!(r"{WB}[A-Za-z0-9._%+-]{{1,64}}@([A-Za-z0-9.-]{{1,253}}\.[A-Za-z]{{2,63}}){WB}")),
            "<redacted:email>@${1}",
        ),
        // JWT: a scan below (`jwt`), in this place of the order.
        (build(&format!(r"{WB}((?i-u:Bearer|Basic))[{s}]{{1,16}}[A-Za-z0-9._~+/=-]{{12,}}")), "${1} <redacted>"),
        (
            build(&format!(r#"([?&](?i-u:token|key|api_key|apikey|secret|password|access_token|sig)=)[^&{s}"']+"#)),
            "${1}<redacted>",
        ),
    ]
});

static HOME_DIR_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let s = js::SPACE_CLASS;
    vec![
        (build(&format!(r#"/Users/[^/{s}"')\]]+"#)), "~"),
        (build(&format!(r#"/home/[^/{s}"')\]]+"#)), "~"),
        (build(&format!(r#"[A-Za-z]:\\(?i-u:Users)\\[^\\{s}"')\]]+"#)), "~"),
    ]
});

/// Conservative on purpose (48+): digests and minified names are shorter, and a git sha (40 hex) stays.
static HIGH_ENTROPY: LazyLock<Regex> = LazyLock::new(|| build(&format!(r"{WB}[A-Za-z0-9_-]{{48,}}{WB}")));
static PEM_BEGIN: LazyLock<Regex> = LazyLock::new(|| build(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----"));
static PEM_END: LazyLock<Regex> = LazyLock::new(|| build(r"-----END [A-Z ]{0,40}PRIVATE KEY-----"));
static TELEGRAM_HEAD: LazyLock<Regex> = LazyLock::new(|| build(&format!(r"{WB}[0-9]{{8,10}}:")));
static JWT_HEAD: LazyLock<Regex> = LazyLock::new(|| build(&format!(r"{WB}eyJ")));

/// Every PEM private key; one left unterminated is cut to the end. A scan, not `BEGIN[\s\S]*?END`: many BEGINs
/// without an END made that quadratic.
fn drop_private_keys(s: &str) -> String {
    if !s.contains("PRIVATE KEY-----") {
        return s.to_string();
    }
    let mut out = String::new();
    let mut from = 0;
    let mut search = 0;
    while let Some(m) = PEM_BEGIN.find_at(s, search) {
        out.push_str(&s[from..m.start()]);
        out.push_str("<redacted:private-key>");
        match PEM_END.find_at(s, m.end()) {
            None => return out,
            Some(end) => {
                from = end.end();
                search = end.end();
            }
        }
    }
    out.push_str(&s[from..]);
    out
}

fn ascii_word(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A Telegram bot token: the bot's number, then exactly 35 characters of `[A-Za-z0-9_-]`.
fn telegram(s: &str) -> String {
    let token = |c: &char| c.is_ascii_alphanumeric() || *c == '_' || *c == '-';
    let mut out = String::new();
    let mut from = 0;
    for m in TELEGRAM_HEAD.find_iter(s) {
        if m.start() < from {
            continue;
        }
        let run = s[m.end()..].chars().take_while(token).count();
        if run == 35 {
            out.push_str(&s[from..m.start()]);
            out.push_str("<redacted:telegram>");
            from = m.end() + 35;
        }
    }
    out.push_str(&s[from..]);
    out
}

/// A JWT: `eyJ` and three dot-separated runs of `[A-Za-z0-9_-]`, the first 8–1024 long and the second 8–8192 (bounded:
/// every `eyJ` in one long run is a start that scans it again; a longer JWT is caught as entropy later).
fn jwt(s: &str) -> String {
    let b = s.as_bytes();
    if !s.contains("eyJ") {
        return s.to_string();
    }
    // How long the run of `[A-Za-z0-9_-]` from each byte is: one pass, so a string of many starts stays linear.
    let mut runs = vec![0usize; b.len() + 1];
    for i in (0..b.len()).rev() {
        let c = b[i];
        runs[i] = if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' { runs[i + 1] + 1 } else { 0 };
    }
    let run = |at: usize| runs[at];
    let mut out = String::new();
    let mut from = 0;
    for m in JWT_HEAD.find_iter(s) {
        if m.start() < from {
            continue;
        }
        let one = run(m.end());
        if !(8..=1024).contains(&one) || b.get(m.end() + one) != Some(&b'.') {
            continue;
        }
        let at2 = m.end() + one + 1;
        let two = run(at2);
        if !(8..=8192).contains(&two) || b.get(at2 + two) != Some(&b'.') {
            continue;
        }
        let at3 = at2 + two + 1;
        let three = run(at3);
        if three < 8 {
            continue;
        }
        out.push_str(&s[from..m.start()]);
        out.push_str("<redacted:jwt>");
        from = at3 + three;
    }
    out.push_str(&s[from..]);
    out
}

/// `scheme://user:password@host`: the password runs to the LAST `@` (within 256 UTF-16 units) that a host follows,
/// so `p@ss` and `pa/ss` go whole. The reference's
/// `\b([a-z][a-z0-9+.-]{0,31}:\/\/)([^\s/:@]{0,256}):([^\s]{1,256})@(?=[^\s@/?#]{0,253}(?:[/?#\s]|$))`, scanned.
fn url_credentials(s: &str) -> String {
    if !s.contains("://") {
        return s.to_string();
    }
    let cs: Vec<(usize, char)> = s.char_indices().collect();
    let n = cs.len();
    let units = |i: usize| cs[i].1.len_utf16();
    // From each place, how many UTF-16 units of host characters follow (`[^\s@/?#]`), and what ends them.
    let mut host = vec![0usize; n + 1];
    for i in (0..n).rev() {
        let c = cs[i].1;
        host[i] = if js::is_space(c) || matches!(c, '@' | '/' | '?' | '#') { 0 } else { host[i + 1] + units(i) };
    }
    let host_follows = |q: usize| -> bool {
        if host[q] > 253 {
            return false;
        }
        let mut k = q;
        while k < n && host[k] > 0 {
            k += 1;
        }
        k == n || matches!(cs[k].1, '/' | '?' | '#') || js::is_space(cs[k].1)
    };
    let mut out = String::new();
    let mut from_byte = 0;
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let prev = if i > 0 { Some(cs[i - 1].1) } else { None };
        if !c.is_ascii_alphabetic() || ascii_word(prev) {
            i += 1;
            continue;
        }
        let scheme_class = |c: char| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-');
        // No further than one past the longest scheme (1 + 31): scanned to the end of the run, every place in a long
        // run of scheme characters (`a-a-a-…`) read the rest of it, quadratic before the length was checked.
        let mut j = i + 1;
        while j < n && j < i + 33 && scheme_class(cs[j].1) {
            j += 1;
        }
        let matched = (|| -> Option<(usize, usize)> {
            if j - i - 1 > 31 || !(j + 2 < n && cs[j].1 == ':' && cs[j + 1].1 == '/' && cs[j + 2].1 == '/') {
                return None;
            }
            let user_start = j + 3;
            let mut k = user_start;
            let mut user_units = 0;
            while k < n && user_units <= 256 && !(js::is_space(cs[k].1) || matches!(cs[k].1, '/' | ':' | '@')) {
                user_units += units(k);
                k += 1;
            }
            if user_units > 256 || k >= n || cs[k].1 != ':' {
                return None;
            }
            let pw_start = k + 1;
            let mut best = None;
            let mut used = 0;
            let mut p = pw_start;
            while p < n && !js::is_space(cs[p].1) {
                if p > pw_start && cs[p].1 == '@' && host_follows(p + 1) {
                    best = Some(p);
                }
                used += units(p);
                if used > 256 {
                    break;
                }
                p += 1;
            }
            best.map(|at| (k, at))
        })();
        match matched {
            Some((colon, at)) => {
                out.push_str(&s[from_byte..cs[colon].0]);
                out.push_str(":<redacted>@");
                from_byte = if at + 1 < n { cs[at + 1].0 } else { s.len() };
                i = at + 1;
            }
            None => i += 1,
        }
    }
    out.push_str(&s[from_byte..]);
    out
}

/// Scrub one string. Never fails; returns `""` for an empty one.
pub fn redact_string(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut out = drop_private_keys(input);
    for (n, (re, replacement)) in SECRET_PATTERNS.iter().enumerate() {
        // The scans take the places the reference's look-ahead patterns hold in its order.
        if n == 12 {
            out = telegram(&out);
            out = url_credentials(&out);
        }
        if n == 15 {
            out = jwt(&out);
        }
        out = re.replace_all(&out, *replacement).into_owned();
    }
    out = KEY_VALUE
        .replace_all(&out, |c: &Captures| {
            let name = c.get(1).map_or("", |m| m.as_str());
            let sep = c.get(2).map_or("", |m| m.as_str());
            match c.get(3) {
                Some(quote) => format!("{name}{sep}{}<redacted>", quote.as_str()),
                None => format!("{name}{sep}{}<redacted>", c.get(5).map_or("", |m| m.as_str())),
            }
        })
        .into_owned();
    for (re, replacement) in HOME_DIR_PATTERNS.iter() {
        out = re.replace_all(&out, *replacement).into_owned();
    }
    HIGH_ENTROPY.replace_all(&out, "<redacted:entropy>").into_owned()
}
