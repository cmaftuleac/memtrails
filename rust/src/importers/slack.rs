//! A Slack workspace export, unzipped: `users.json`, `channels.json` (and `groups.json`, `mpims.json`, `dms.json` when
//! present), and a folder a conversation holding one `YYYY-MM-DD.json` a day. Read a day at a time, in order.
//!
//! One record a message that says something. id: `channel id:ts`. text: the message with `<@U123>` as `@Name`,
//! `<#C1|name>` as `#name`, `<url|label>` as `label (url)`, then the names of its files and attachments. author: the
//! person's real name, else display name, else user name (a bot: its name). at: the message's ts. thread: `channel:ts`
//! of the thread's first message (a message outside a thread starts its own). collection: the option's or the
//! channel's name. kind `message`. Joins, leaves, topic changes and the like are left out.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::{Captures, Regex};
use serde_json::Value;

use super::text::{js_number_of, js_string, re, resolve_path, truthy, Draft};
use super::Records;
use crate::error::{Error, Result};
use crate::js;
use crate::records::{At, MemoryRecord};

/// Message subtypes left out by default: events about the channel, not things said in it.
pub const SLACK_SKIPPED_SUBTYPES: &[&str] = &[
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "channel_archive",
    "channel_unarchive",
    "channel_convert_to_private",
    "channel_convert_to_public",
    "group_join",
    "group_leave",
    "group_topic",
    "group_purpose",
    "group_name",
    "group_archive",
    "group_unarchive",
    "pinned_item",
    "unpinned_item",
    "bot_add",
    "bot_remove",
    "tombstone",
    "reminder_add",
    "joiner_notification",
    "joiner_notification_for_inviter",
    "message_deleted",
    "sh_room_created",
    "channel_canvas_updated",
];

/// How an export is read.
#[derive(Debug, Clone, Default)]
pub struct SlackOptions {
    /// Only these conversations (by name or id; a leading `#` is dropped).
    pub channels: Option<Vec<String>>,
    /// Default: the channel's name.
    pub collection: Option<String>,
    /// Default: the export folder's name.
    pub source: Option<String>,
    /// Subtypes to leave out. Default: `SLACK_SKIPPED_SUBTYPES`.
    pub skip_subtypes: Option<Vec<String>>,
    /// Leave out messages posted by bots and integrations.
    pub skip_bots: bool,
    /// The workspace's address (`https://example.slack.com`), for a link to each message.
    pub workspace_url: Option<String>,
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::Import(format!("cannot read {}: {e}", path.display()))),
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    serde_json::from_str(text).map(Some).map_err(|e| Error::Import(format!("cannot read {}: {e}", path.display())))
}

fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn person_name(u: &Value) -> Option<String> {
    let profile = u.get("profile");
    [
        profile.and_then(|p| str_of(p, "real_name")),
        str_of(u, "real_name"),
        profile.and_then(|p| str_of(p, "display_name")),
        str_of(u, "display_name"),
        str_of(u, "name"),
    ]
    .into_iter()
    .flatten()
    .map(js::trim)
    .find(|n| !n.is_empty())
    .map(str::to_string)
}

static MARKUP: LazyLock<Regex> = LazyLock::new(|| re(r"<([^<>\n]*)>"));
static SPECIAL: LazyLock<Regex> = LazyLock::new(|| re(r"^(here|channel|everyone)$"));

/// Slack's markup as plain text: mentions as names, channel links as `#name`, links as `label (url)`.
pub fn slack_text(text: &str, users: &HashMap<String, String>, channels: &HashMap<String, String>) -> String {
    let replaced = MARKUP.replace_all(text, |c: &Captures| {
        let inner = &c[1];
        let (target, label) = match inner.find('|') {
            Some(bar) => (&inner[..bar], &inner[bar + 1..]),
            None => (inner, ""),
        };
        if let Some(id) = target.strip_prefix('@') {
            let name = users.get(id).cloned().unwrap_or_else(|| {
                let l = label.strip_prefix('@').unwrap_or(label);
                if l.is_empty() {
                    id.to_string()
                } else {
                    l.to_string()
                }
            });
            return format!("@{name}");
        }
        if let Some(id) = target.strip_prefix('#') {
            let name = if !label.is_empty() {
                label.to_string()
            } else {
                channels.get(id).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| id.to_string())
            };
            return format!("#{name}");
        }
        if let Some(special) = target.strip_prefix('!') {
            if SPECIAL.is_match(special) {
                return format!("@{special}");
            }
            if !label.is_empty() {
                return label.to_string();
            }
            // `<!subteam^S123>`: the part before `^` (`\^.*$`, `.` stopping at a line end).
            let cut = match special.find('^') {
                Some(at) if !special[at..].contains(['\r', '\u{2028}', '\u{2029}']) => &special[..at],
                _ => special,
            };
            return format!("@{cut}");
        }
        if target.is_empty() {
            return c[0].to_string();
        }
        if let Some(address) = target.strip_prefix("mailto:") {
            return if label.is_empty() { address.to_string() } else { label.to_string() };
        }
        if !label.is_empty() && label != target {
            format!("{label} ({target})")
        } else {
            target.to_string()
        }
    });
    replaced.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
}

struct Conversation {
    id: String,
    name: String,
    folder: String,
}

struct Export {
    root: PathBuf,
    source: String,
    skip: HashSet<String>,
    users: HashMap<String, String>,
    channel_names: HashMap<String, String>,
    conversations: VecDeque<Conversation>,
    /// The conversation being read, and its days still to read.
    current: Option<(Conversation, PathBuf, VecDeque<String>)>,
    out: VecDeque<MemoryRecord>,
    o: SlackOptions,
}

impl Export {
    /// The next conversation's folder and days, when it is wanted and lies inside the export.
    fn open(&self, c: &Conversation) -> Result<Option<(PathBuf, VecDeque<String>)>> {
        if let Some(wanted) = &self.o.channels {
            let wanted: HashSet<&str> = wanted.iter().map(|w| w.strip_prefix('#').unwrap_or(w)).collect();
            if !wanted.contains(c.name.as_str()) && !wanted.contains(c.id.as_str()) {
                return Ok(None);
            }
        }
        // The folder's name comes from the export's own lists: `../x`, `a/b` or an absolute path must not lead out
        // of it.
        if c.folder.is_empty() {
            return Ok(None);
        }
        let folder = resolve_path(&self.root.join(&c.folder)).map_err(|e| Error::io(&self.root, e))?;
        if folder.parent() != Some(self.root.as_path()) || !fs::metadata(&folder).is_ok_and(|m| m.is_dir()) {
            return Ok(None);
        }
        let mut days: Vec<String> = fs::read_dir(&folder)
            .map_err(|e| Error::io(&folder, e))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|f| is_day_file(f))
            .collect();
        days.sort();
        Ok(Some((folder, days.into())))
    }

    fn read_day(&mut self, c: &Conversation, file: &Path) -> Result<()> {
        let Some(Value::Array(messages)) = read_json(file)? else { return Ok(()) };
        for m in &messages {
            if let Some(r) = self.message(c, m) {
                self.out.push_back(r);
            }
        }
        Ok(())
    }

    fn message(&self, c: &Conversation, m: &Value) -> Option<MemoryRecord> {
        if !m.is_object() || !m.get("ts").is_some_and(truthy) {
            return None;
        }
        if m.get("type").is_some_and(truthy) && str_of(m, "type") != Some("message") {
            return None;
        }
        if m.get("hidden").is_some_and(truthy) {
            return None;
        }
        let subtype = str_of(m, "subtype").filter(|s| !s.is_empty());
        if subtype.is_some_and(|s| self.skip.contains(s)) {
            return None;
        }
        let user = str_of(m, "user").filter(|s| !s.is_empty());
        let bot = subtype == Some("bot_message") || (m.get("bot_id").is_some_and(truthy) && user.is_none());
        if bot && self.o.skip_bots {
            return None;
        }
        let ts = js_string(m.get("ts")?);
        let mut extras: Vec<String> = Vec::new();
        for f in m.get("files").and_then(Value::as_array).into_iter().flatten() {
            let t = str_of(f, "title").filter(|t| !t.is_empty()).or_else(|| str_of(f, "name"));
            if let Some(t) = t.filter(|t| !js::trim(t).is_empty()) {
                extras.push(format!("File: {t}"));
            }
        }
        for a in m.get("attachments").and_then(Value::as_array).into_iter().flatten() {
            let Some(t) = str_of(a, "title").filter(|t| !t.is_empty()).or_else(|| str_of(a, "fallback")) else {
                continue;
            };
            if js::trim(t).is_empty() {
                continue;
            }
            let link = str_of(a, "title_link").filter(|l| !l.is_empty()).or_else(|| str_of(a, "from_url"));
            let shown = match link.filter(|l| !l.is_empty() && !t.contains(l)) {
                Some(l) => format!(" ({l})"),
                None => String::new(),
            };
            extras.push(format!("Attachment: {}{shown}", slack_text(t, &self.users, &self.channel_names)));
        }
        let said = slack_text(str_of(m, "text").unwrap_or(""), &self.users, &self.channel_names);
        let said = js::trim(&said);
        let text =
            std::iter::once(said.to_string()).chain(extras).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n");
        if text.is_empty() {
            return None;
        }
        let author = user
            .and_then(|u| self.users.get(u).cloned())
            .or_else(|| m.get("user_profile").filter(|p| truthy(p)).and_then(person_name))
            .or_else(|| {
                m.get("bot_profile").and_then(|b| str_of(b, "name")).filter(|n| !n.is_empty()).map(str::to_string)
            })
            .or_else(|| str_of(m, "username").filter(|n| !n.is_empty()).map(str::to_string))
            .or_else(|| user.map(str::to_string));
        let ms = (js_number_of(&ts) * 1000.0 + 0.5).floor();
        let thread_ts = m.get("thread_ts").filter(|v| !v.is_null()).map(js_string).unwrap_or_else(|| ts.clone());
        let url = self
            .o
            .workspace_url
            .as_ref()
            .map(|w| format!("{}/archives/{}/p{}", w.trim_end_matches('/'), c.id, ts.replacen('.', "", 1)));
        let field = |k: &str| m.get(k).cloned();
        Some(
            Draft {
                id: format!("{}:{ts}", c.id),
                text,
                kind: Some("message".into()),
                source: Some(self.source.clone()),
                author,
                collection: Some(self.o.collection.clone().unwrap_or_else(|| c.name.clone())),
                thread: Some(format!("{}:{thread_ts}", c.name)),
                at: ms.is_finite().then_some(At::Ms(ms)),
                url,
                metadata: vec![
                    ("channel".into(), Some(Value::String(c.name.clone()))),
                    ("ts".into(), field("ts")),
                    ("threadTs".into(), field("thread_ts")),
                    ("subtype".into(), field("subtype")),
                    ("replies".into(), field("reply_count")),
                ],
                ..Default::default()
            }
            .finish(),
        )
    }
}

/// `^\d{4}-\d{2}-\d{2}\.json$`.
fn is_day_file(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() == 15
        && name.ends_with(".json")
        && b[..10].iter().enumerate().all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() })
}

impl Iterator for Export {
    type Item = Result<MemoryRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(r) = self.out.pop_front() {
                return Some(Ok(r));
            }
            if let Some((c, folder, days)) = &mut self.current {
                if let Some(day) = days.pop_front() {
                    let file = folder.join(day);
                    let c = Conversation { id: c.id.clone(), name: c.name.clone(), folder: c.folder.clone() };
                    if let Err(e) = self.read_day(&c, &file) {
                        return Some(Err(e));
                    }
                    continue;
                }
                self.current = None;
            }
            let c = self.conversations.pop_front()?;
            match self.open(&c) {
                Err(e) => return Some(Err(e)),
                Ok(None) => {}
                Ok(Some((folder, days))) => self.current = Some((c, folder, days)),
            }
        }
    }
}

/// Records from an unzipped Slack export folder.
pub fn from_slack_export(dir: impl AsRef<Path>, opts: SlackOptions) -> Result<Records> {
    let root = resolve_path(dir.as_ref()).map_err(|e| Error::io(dir.as_ref(), e))?;
    let source = opts
        .source
        .clone()
        .unwrap_or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let skip: HashSet<String> = opts
        .skip_subtypes
        .clone()
        .unwrap_or_else(|| SLACK_SKIPPED_SUBTYPES.iter().map(|s| s.to_string()).collect())
        .into_iter()
        .collect();
    let mut users = HashMap::new();
    if let Some(Value::Array(list)) = read_json(&root.join("users.json"))? {
        for u in &list {
            if let (Some(id), Some(n)) = (str_of(u, "id").filter(|i| !i.is_empty()), person_name(u)) {
                users.insert(id.to_string(), n);
            }
        }
    }
    // Each conversation: its id, its name, and the folder its days are in (a direct message's folder is its id).
    let mut conversations: VecDeque<Conversation> = VecDeque::new();
    for (file, by_id) in [("channels.json", false), ("groups.json", false), ("mpims.json", false), ("dms.json", true)] {
        let Some(Value::Array(list)) = read_json(&root.join(file))? else { continue };
        for c in &list {
            let Some(id) = str_of(c, "id").filter(|i| !i.is_empty()) else { continue };
            let channel_name = str_of(c, "name").filter(|n| !n.is_empty());
            let name = if by_id {
                let members: Vec<String> = c
                    .get("members")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|m| {
                        let key = js_string(m);
                        users.get(&key).cloned().unwrap_or(key)
                    })
                    .collect();
                let joined = members.join(", ");
                if joined.is_empty() {
                    id.to_string()
                } else {
                    joined
                }
            } else {
                channel_name.unwrap_or(id).to_string()
            };
            let folder = if by_id { id.to_string() } else { channel_name.unwrap_or(id).to_string() };
            conversations.push_back(Conversation { id: id.to_string(), name, folder });
        }
    }
    if conversations.is_empty() {
        // No lists: every folder is a conversation named after it.
        let mut names: Vec<String> = fs::read_dir(&root)
            .map_err(|e| Error::io(&root, e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.starts_with('.'))
            .collect();
        names.sort();
        for n in names {
            conversations.push_back(Conversation { id: n.clone(), name: n.clone(), folder: n });
        }
    }
    let channel_names = conversations.iter().map(|c| (c.id.clone(), c.name.clone())).collect();
    Ok(Box::new(Export {
        root,
        source,
        skip,
        users,
        channel_names,
        conversations,
        current: None,
        out: VecDeque::new(),
        o: opts,
    }))
}
