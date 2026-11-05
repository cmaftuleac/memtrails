//! What JavaScript does that the reference relies on, done the same way here: strings measured in UTF-16 code units,
//! JavaScript's whitespace, `JSON.stringify`'s numbers, `Date.parse` and `toISOString`. Every module that measures,
//! cuts, hashes or dates text goes through these, so a memory written by either library reads the same in the other.

use chrono::{Local, NaiveDate, NaiveDateTime, TimeZone};
use serde_json::Value;

/// JavaScript's `\s`: the Unicode spaces it lists, U+FEFF included and U+0085 not (Rust's `char::is_whitespace`
/// differs on both).
pub(crate) fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

/// The same set as a regex class body, for patterns that say `\s`.
pub(crate) const SPACE_CLASS: &str =
    r"\t\n\x0B\x0C\r \u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}\u{feff}";

/// `String.prototype.trim`.
pub(crate) fn trim(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `String.prototype.trimEnd`.
pub(crate) fn trim_end(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `s.replace(/\s+/g, ' ')`.
pub(crate) fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_space = false;
    for c in s.chars() {
        if is_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// A string's length as JavaScript counts it: UTF-16 code units.
pub fn len16(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// The byte offset of UTF-16 unit `unit` (clamped to the end). A unit inside a character outside the Basic
/// Multilingual Plane rounds down to that character's start: a Rust string cannot hold half of it.
pub(crate) fn byte_at16(s: &str, unit: usize) -> usize {
    let mut at = 0usize;
    for (b, c) in s.char_indices() {
        let next = at + c.len_utf16();
        if next > unit {
            return b;
        }
        at = next;
    }
    s.len()
}

/// `s.slice(a, b)` in UTF-16 units (both clamped; an empty string when `b <= a`).
pub(crate) fn slice16(s: &str, a: usize, b: usize) -> &str {
    if b <= a {
        return "";
    }
    let start = byte_at16(s, a);
    let end = byte_at16(s, b);
    if end <= start {
        ""
    } else {
        &s[start..end]
    }
}

/// `s.slice(0, n)` in UTF-16 units.
pub(crate) fn head16(s: &str, n: usize) -> &str {
    &s[..byte_at16(s, n)]
}

/// `s.slice(-n)` in UTF-16 units (`s.slice(-0)` is the whole string, as in JavaScript).
pub(crate) fn tail16(s: &str, n: usize) -> &str {
    if n == 0 {
        return s;
    }
    let len = len16(s);
    if n >= len {
        return s;
    }
    &s[byte_at16(s, len - n)..]
}

/// A number as JavaScript writes it (`String(n)`, and inside `JSON.stringify`): integers without a fraction,
/// exponents from 1e21 and below 1e-6.
pub fn number(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    // The shortest digits that read back as `x`, and where the point goes.
    let e = format!("{:e}", x.abs());
    let (mantissa, exp) = e.split_once('e').unwrap_or((e.as_str(), "0"));
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let exp: i32 = exp.parse().unwrap_or(0);
    let k = digits.len() as i32;
    let n = exp + 1;
    let sign = if x < 0.0 { "-" } else { "" };
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let rest = if k > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let e = n - 1;
        format!("{}{rest}e{}{}", &digits[..1], if e >= 0 { "+" } else { "-" }, e.abs())
    };
    format!("{sign}{body}")
}

/// `JSON.stringify(v)`: compact, keys in their order, numbers as JavaScript writes them.
pub fn stringify(v: &Value) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn write_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&number(i as f64));
            } else if let Some(u) = n.as_u64() {
                out.push_str(&number(u as f64));
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                out.push_str(&if f.is_finite() { number(f) } else { "null".to_string() });
            }
        }
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            // In JavaScript's property order: integer-like keys first, ascending, then the rest as they came. A
            // record's metadata with a key like "2" fingerprinted otherwise than the reference's.
            let mut keys: Vec<(&String, &Value)> = o.iter().collect();
            if keys.iter().any(|(k, _)| array_index(k).is_some()) {
                keys.sort_by_key(|(k, _)| array_index(k).map_or((1, 0), |n| (0, n)));
            }
            for (i, (k, x)) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
                out.push(':');
                write_json(x, out);
            }
            out.push('}');
        }
    }
}

/// A key JavaScript orders as an integer (an array index): digits without a leading zero (but "0"), below 2³² − 1.
fn array_index(k: &str) -> Option<u32> {
    if k.is_empty() || k.len() > 10 || !k.bytes().all(|b| b.is_ascii_digit()) || (k.len() > 1 && k.starts_with('0')) {
        return None;
    }
    k.parse::<u32>().ok().filter(|n| *n < u32::MAX)
}

/// An object's entries in JavaScript's property order: integer-like keys first, ascending, then the rest as they came
/// — the order the reference's objects iterate and `JSON.stringify` writes them in.
pub(crate) fn in_js_order<V: Clone>(m: &indexmap::IndexMap<String, V>) -> indexmap::IndexMap<String, V> {
    let mut entries: Vec<(&String, &V)> = m.iter().collect();
    entries.sort_by_key(|(k, _)| array_index(k).map_or((1, 0), |n| (0, n)));
    entries.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// A whole number of milliseconds as a JSON number when it is one (so a log line reads `1775088000000`, as
/// JavaScript writes it, not `1775088000000.0`).
pub fn ms_value(ms: f64) -> Value {
    if ms.fract() == 0.0 && ms.abs() <= 9_007_199_254_740_991.0 {
        Value::from(ms as i64)
    } else {
        serde_json::Number::from_f64(ms).map(Value::Number).unwrap_or(Value::Null)
    }
}

const DAY_MS: f64 = 86_400_000.0;

/// Days since 1970-01-01 of a proleptic Gregorian date; month 0–11, any day (it rolls over, as `Date.UTC` does).
pub(crate) fn days_from_civil(y: i64, m0: i64, d: i64) -> i64 {
    let y = y + m0.div_euclid(12);
    let m = m0.rem_euclid(12) + 1;
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468 + (d - 1)
}

/// `Date.UTC(y, m0, d, h, mi, s, ms)`.
pub(crate) fn date_utc(y: i64, m0: i64, d: i64, h: i64, mi: i64, s: i64, ms: i64) -> f64 {
    days_from_civil(y, m0, d) as f64 * DAY_MS + ((h * 60 + mi) * 60 + s) as f64 * 1000.0 + ms as f64
}

/// A time's UTC calendar fields: year, month (1–12), day, hours, minutes, seconds, milliseconds, weekday (0 = Sunday).
pub(crate) struct Utc {
    pub year: i64,
    pub month: i64,
    pub day: i64,
    pub hour: i64,
    pub minute: i64,
    pub second: i64,
    pub milli: i64,
    pub weekday: i64,
}

pub(crate) fn utc_of(ms: f64) -> Utc {
    let t = ms.floor() as i64;
    let days = t.div_euclid(86_400_000);
    let rest = t.rem_euclid(86_400_000);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    Utc {
        year,
        month,
        day,
        hour: rest / 3_600_000,
        minute: rest / 60_000 % 60,
        second: rest / 1000 % 60,
        milli: rest % 1000,
        weekday: (days + 4).rem_euclid(7),
    }
}

/// `new Date(ms).toISOString()`.
pub fn iso_string(ms: f64) -> String {
    let u = utc_of(ms);
    let year = if (0..=9999).contains(&u.year) {
        format!("{:04}", u.year)
    } else {
        format!("{}{:06}", if u.year < 0 { "-" } else { "+" }, u.year.abs())
    };
    format!("{year}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", u.month, u.day, u.hour, u.minute, u.second, u.milli)
}

/// The date of `ms` as `YYYY-MM-DD` (`toISOString().slice(0, 10)`).
pub(crate) fn iso_day(ms: f64) -> String {
    let s = iso_string(ms);
    s[..s.len() - 14].to_string()
}

pub(crate) const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
pub(crate) const MONTHS: [&str; 12] =
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// `toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' })`.
pub(crate) fn long_day(ms: f64) -> String {
    let u = utc_of(ms);
    format!("{} {} {}", u.day, MONTH_NAMES[(u.month - 1) as usize], u.year)
}

/// `a.localeCompare(b)` as ICU's root collation orders most text: by the letters first (case and accents aside), then
/// by accents, then lower case before upper case.
pub(crate) fn locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    crate::text::fold(a).cmp(&crate::text::fold(b)).then_with(|| a.to_lowercase().cmp(&b.to_lowercase())).then_with(
        || {
            for (x, y) in a.chars().zip(b.chars()) {
                if x != y {
                    return match (x.is_lowercase(), y.is_lowercase()) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => x.cmp(&y),
                    };
                }
            }
            a.len().cmp(&b.len())
        },
    )
}

/// Local wall-clock fields as milliseconds since the epoch, the way `new Date(y, m, d, …)` reads them: in the
/// machine's time zone; a time the clocks skipped is read with the offset before the change.
fn local_ms(y: i64, m0: i64, d: i64, h: i64, mi: i64, s: i64, ms: i64) -> Option<f64> {
    let as_utc = date_utc(y, m0, d, h, mi, s, ms);
    let naive = utc_naive(as_utc)?;
    let offset_of = |n: &NaiveDateTime| -> Option<i64> {
        Local.from_local_datetime(n).earliest().map(|t| i64::from(t.offset().local_minus_utc()))
    };
    let offset = match offset_of(&naive) {
        Some(o) => o,
        None => offset_of(&(naive - chrono::Duration::hours(3)))?,
    };
    Some(as_utc - offset as f64 * 1000.0)
}

fn utc_naive(ms: f64) -> Option<NaiveDateTime> {
    let u = utc_of(ms);
    NaiveDate::from_ymd_opt(i32::try_from(u.year).ok()?, u.month as u32, u.day as u32)?.and_hms_milli_opt(
        u.hour as u32,
        u.minute as u32,
        u.second as u32,
        u.milli as u32,
    )
}

const MAX_TIME: f64 = 8.64e15;

fn finite_time(t: f64) -> Option<f64> {
    (t.is_finite() && t.abs() <= MAX_TIME).then_some(t)
}

/// `Date.parse(s)` as V8 reads it: ISO 8601 first (a date alone is UTC midnight, a time without an offset is local),
/// then the legacy forms — RFC 2822 (`Tue, 2 Apr 2024 10:30:00 +0000`), `asctime`, `April 2026`, `2024/04/02` — and
/// `None` for anything else.
pub fn date_parse(s: &str) -> Option<f64> {
    if let Some(t) = parse_iso(s) {
        return finite_time(t);
    }
    parse_legacy(s).and_then(finite_time)
}

fn digits(b: &[u8], at: usize, n: usize) -> Option<i64> {
    if at + n > b.len() || !b[at..at + n].iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(&b[at..at + n]).ok()?.parse().ok()
}

fn parse_iso(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i;
    let year = match b.first() {
        Some(b'+') | Some(b'-') => {
            let y = digits(b, 1, 6)?;
            i = 7;
            if b[0] == b'-' {
                if y == 0 {
                    return None;
                }
                -y
            } else {
                y
            }
        }
        _ => {
            let y = digits(b, 0, 4)?;
            i = 4;
            y
        }
    };
    let (mut month, mut day) = (1, 1);
    if b.get(i) == Some(&b'-') {
        month = digits(b, i + 1, 2)?;
        i += 3;
        if b.get(i) == Some(&b'-') {
            day = digits(b, i + 1, 2)?;
            i += 3;
        }
    }
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if i == b.len() {
        return Some(date_utc(year, month - 1, day, 0, 0, 0, 0));
    }
    if b[i] != b'T' && b[i] != b't' {
        return None;
    }
    let hour = digits(b, i + 1, 2)?;
    if b.get(i + 3) != Some(&b':') {
        return None;
    }
    let minute = digits(b, i + 4, 2)?;
    i += 6;
    let (mut second, mut milli) = (0, 0);
    if b.get(i) == Some(&b':') {
        second = digits(b, i + 1, 2)?;
        i += 3;
        if b.get(i) == Some(&b'.') {
            let start = i + 1;
            let mut end = start;
            while end < b.len() && b[end].is_ascii_digit() {
                end += 1;
            }
            if end == start {
                return None;
            }
            let frac = &s[start..end.min(start + 3)];
            milli = format!("{frac:0<3}").parse().ok()?;
            i = end;
        }
    }
    if hour > 24 || minute > 59 || second > 59 || (hour == 24 && (minute > 0 || second > 0 || milli > 0)) {
        return None;
    }
    if i == b.len() {
        return local_ms(year, month - 1, day, hour, minute, second, milli);
    }
    let offset = match b[i] {
        b'Z' | b'z' if i + 1 == b.len() => 0,
        b'+' | b'-' => {
            let h = digits(b, i + 1, 2)?;
            let m = if b.get(i + 3) == Some(&b':') && i + 6 == b.len() {
                digits(b, i + 4, 2)?
            } else {
                return None;
            };
            if h > 23 || m > 59 {
                return None;
            }
            let o = h * 60 + m;
            if b[i] == b'-' {
                -o
            } else {
                o
            }
        }
        _ => return None,
    };
    Some(date_utc(year, month - 1, day, hour, minute, second, milli) - offset as f64 * 60_000.0)
}

/// US time zones as RFC 2822 names them, in minutes from UTC.
fn zone(word: &str) -> Option<i64> {
    Some(match word {
        "ut" | "utc" | "gmt" | "z" => 0,
        "edt" => -4 * 60,
        "est" | "cdt" => -5 * 60,
        "cst" | "mdt" => -6 * 60,
        "mst" | "pdt" => -7 * 60,
        "pst" => -8 * 60,
        _ => return None,
    })
}

enum Tok {
    Num(i64, usize),
    Word(String),
    Sym(char),
    Space,
}

fn tokens(s: &str) -> Option<Vec<Tok>> {
    let mut out = Vec::new();
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_ascii_digit() {
            let start = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            // As V8 reads a number: leading zeros skipped, then nine significant digits at most — the rest counted in
            // its length, not read. So no number here is large enough to overflow what is done with it.
            let n = cs[start..i]
                .iter()
                .skip_while(|c| **c == '0')
                .take(9)
                .fold(0i64, |v, c| v * 10 + i64::from(c.to_digit(10).unwrap_or(0)));
            out.push(Tok::Num(n, i - start));
        } else if c.is_alphabetic() {
            let start = i;
            while i < cs.len() && cs[i].is_alphabetic() {
                i += 1;
            }
            out.push(Tok::Word(cs[start..i].iter().collect::<String>().to_lowercase()));
        } else if c == '(' {
            // A comment, nested as RFC 2822 allows.
            let mut depth = 0;
            while i < cs.len() {
                if cs[i] == '(' {
                    depth += 1;
                } else if cs[i] == ')' {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                i += 1;
            }
            out.push(Tok::Space);
        } else if is_space(c) {
            while i < cs.len() && is_space(cs[i]) {
                i += 1;
            }
            out.push(Tok::Space);
        } else {
            out.push(Tok::Sym(c));
            i += 1;
        }
    }
    Some(out)
}

/// V8's fallback date reader, for the forms people and mail programs write.
fn parse_legacy(s: &str) -> Option<f64> {
    let toks = tokens(s)?;
    let mut date: Vec<(i64, usize)> = Vec::new();
    let mut named_month: Option<i64> = None;
    let mut time: Vec<i64> = Vec::new();
    let mut milli = 0i64;
    let mut pm: Option<bool> = None;
    let mut tz: Option<i64> = None; // seconds east of UTC
    let mut seen_number = false;
    let mut i = 0;
    let at = |i: usize| toks.get(i);
    while i < toks.len() {
        match &toks[i] {
            Tok::Num(n, len) => {
                seen_number = true;
                if matches!(at(i + 1), Some(Tok::Sym(':'))) && time.is_empty() {
                    // hh:mm[:ss[.sss]]
                    time.push(*n);
                    i += 2;
                    while let Some(Tok::Num(v, _)) = at(i) {
                        time.push(*v);
                        i += 1;
                        if matches!(at(i), Some(Tok::Sym(':'))) && time.len() < 3 {
                            i += 1;
                            continue;
                        }
                        if matches!(at(i), Some(Tok::Sym('.'))) && time.len() == 3 {
                            if let Some(Tok::Num(f, flen)) = at(i + 1) {
                                let mut v = *f;
                                let mut l = (*flen).min(9); // the digits V8 read
                                while l > 3 {
                                    v /= 10;
                                    l -= 1;
                                }
                                while l < 3 {
                                    v *= 10;
                                    l += 1;
                                }
                                milli = v;
                                i += 2;
                            }
                        }
                        break;
                    }
                    if time.len() < 2 {
                        return None;
                    }
                    continue;
                }
                if date.len() >= 3 {
                    return None;
                }
                date.push((*n, *len));
                i += 1;
                // A date's own separator: `2026-04-02` after a time is a date, not an offset.
                if matches!(at(i), Some(Tok::Sym('-'))) {
                    i += 1;
                }
            }
            Tok::Sym(c @ ('+' | '-')) if !time.is_empty() || tz.is_some() => {
                // An offset after the time: +hhmm, +hh:mm or +hh — and no other length.
                let sign = if *c == '-' { -1 } else { 1 };
                let Some(Tok::Num(n, len)) = at(i + 1) else { return None };
                let (h, m) = if matches!(at(i + 2), Some(Tok::Sym(':'))) {
                    let Some(Tok::Num(m, _)) = at(i + 3) else { return None };
                    i += 4;
                    (*n, *m)
                } else {
                    i += 2;
                    match len {
                        1 | 2 => (*n, 0),
                        3 | 4 => (n / 100, n % 100),
                        _ => return None,
                    }
                };
                // In seconds, in 32 bits as V8 counts them: one past its small-integer range is no date, one within it
                // that wrapped is the date V8 gives.
                let secs = (h as u32).wrapping_mul(3600).wrapping_add((m as u32).wrapping_mul(60));
                if secs > i32::MAX as u32 {
                    return None;
                }
                tz = Some(tz.unwrap_or(0).checked_add(sign * i64::from(secs))?);
            }
            Tok::Sym('-' | '/' | ',' | '.') | Tok::Space => i += 1,
            Tok::Sym(_) => return None,
            Tok::Word(w) => {
                i += 1;
                if w == "am" || w == "pm" {
                    pm = Some(w == "pm");
                } else if let Some(z) = zone(w) {
                    tz = Some(z * 60);
                } else if w == "t" {
                    // Only between a date and its time.
                    let time_next = matches!((at(i), at(i + 1)), (Some(Tok::Num(..)), Some(Tok::Sym(':'))));
                    if !seen_number || !time_next {
                        return None;
                    }
                } else if w.len() >= 3 {
                    if let Some(m) = MONTH_NAMES
                        .iter()
                        .position(|n| n.to_lowercase().starts_with(w.as_str()) || w.starts_with(&n[..3].to_lowercase()))
                    {
                        if named_month.is_some() {
                            return None;
                        }
                        named_month = Some(m as i64);
                        continue;
                    }
                    if WEEKDAYS.iter().any(|d| w.starts_with(&d.to_lowercase())) {
                        continue;
                    }
                    if seen_number {
                        return None;
                    }
                } else if seen_number {
                    return None;
                }
            }
        }
    }
    if date.is_empty() && named_month.is_none() {
        return None;
    }
    let year_of = |(n, len): (i64, usize)| -> i64 {
        if len <= 2 {
            if n < 50 {
                2000 + n
            } else {
                1900 + n
            }
        } else {
            n
        }
    };
    let (year, month0, day) = if let Some(m) = named_month {
        match date.as_slice() {
            [a] => (year_of(*a), m, 1),
            [a, b] => {
                if a.1 >= 3 || a.0 > 31 {
                    (year_of(*a), m, b.0)
                } else {
                    (year_of(*b), m, a.0)
                }
            }
            _ => return None,
        }
    } else {
        match date.as_slice() {
            [a, b, c] if a.1 >= 3 => (year_of(*a), b.0 - 1, c.0),
            [a, b, c] => (year_of(*c), a.0 - 1, b.0),
            [a, b] if a.1 >= 3 => (year_of(*a), b.0 - 1, 1),
            [a, b] => (2001, a.0 - 1, b.0),
            _ => return None,
        }
    };
    if !(0..12).contains(&month0) || !(1..=31).contains(&day) {
        return None;
    }
    let (mut h, mi, sec) =
        (time.first().copied().unwrap_or(0), time.get(1).copied().unwrap_or(0), time.get(2).copied().unwrap_or(0));
    if let Some(p) = pm {
        if h > 12 {
            return None;
        }
        h = if p { h % 12 + 12 } else { h % 12 };
    }
    if h > 24 || mi > 59 || sec > 59 || (h == 24 && (mi > 0 || sec > 0)) {
        return None;
    }
    match tz {
        Some(off) => Some(date_utc(year, month0, day, h, mi, sec, milli) - off as f64 * 1000.0),
        None => local_ms(year, month0, day, h, mi, sec, milli),
    }
}

/// The current time in milliseconds (`Date.now()`).
pub fn now_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_as_javascript_writes_them() {
        assert_eq!(number(1e21), "1e+21");
        assert_eq!(number(1.5e-7), "1.5e-7");
        assert_eq!(number(0.1), "0.1");
        assert_eq!(number(-0.0), "0");
        assert_eq!(number(123456789012345680000.0), "123456789012345680000");
        assert_eq!(number(1e-6), "0.000001");
        assert_eq!(number(1775088000000.0), "1775088000000");
        assert_eq!(number(-2.5), "-2.5");
    }

    #[test]
    fn dates_as_v8_reads_them() {
        assert_eq!(date_parse("2024-02-30"), Some(1709251200000.0));
        assert_eq!(date_parse("2024-02-32"), None);
        assert_eq!(date_parse("2026-04-02T24:00:00Z"), Some(1775174400000.0));
        assert_eq!(date_parse("2026-04-02T10:30:00.1234567Z"), Some(1775125800123.0));
        assert_eq!(date_parse("+002026-04-02"), Some(1775088000000.0));
        assert_eq!(date_parse("2026-04-02t10:30:00z"), Some(1775125800000.0));
        assert_eq!(date_parse("Tue, 02 Apr 2024 10:30:00 +0000"), Some(1712053800000.0));
        assert_eq!(date_parse("2 Apr 2024 10:30 GMT"), Some(1712053800000.0));
        assert_eq!(date_parse("Tue, 2 Apr 2024 10:30:00 PDT"), Some(1712079000000.0));
        assert_eq!(date_parse("Tue, 02 Apr 2024 10:30:00 +0000 (UTC)"), Some(1712053800000.0));
        assert_eq!(date_parse("02 Apr 24 10:30 +0000"), Some(1712053800000.0));
        assert_eq!(date_parse("2026-04-02T10:30:00+0300"), Some(1775115000000.0));
        assert_eq!(date_parse("20260402"), None);
        assert_eq!(date_parse("not a date"), None);
        assert_eq!(date_parse("yesterday-ish"), None);
        assert!(date_parse("April 2026").is_some());
        assert!(date_parse("Mon Jan 1 00:00:00 2024").is_some());
    }

    #[test]
    fn utf16_slices() {
        let s = "a🚀b";
        assert_eq!(len16(s), 4);
        assert_eq!(slice16(s, 1, 3), "🚀");
        assert_eq!(head16(s, 2), "a");
        assert_eq!(iso_string(1775088000000.0), "2026-04-02T00:00:00.000Z");
    }
}
