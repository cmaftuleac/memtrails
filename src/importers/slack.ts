/**
 * A Slack workspace export, unzipped: `users.json`, `channels.json` (and `groups.json`, `mpims.json`, `dms.json` when
 * present), and a folder a conversation holding one `YYYY-MM-DD.json` a day. Read a day at a time, in order.
 *
 * One record a message that says something. id: `channel id:ts`. text: the message with `<@U123>` as `@Name`,
 * `<#C1|name>` as `#name`, `<url|label>` as `label (url)`, then the names of its files and attachments. author: the
 * person's real name, else display name, else user name (a bot: its name). at: the message's ts. thread: `channel:ts`
 * of the thread's first message (a message outside a thread starts its own). collection: `opts.collection` or the
 * channel's name. kind `message`. Joins, leaves, topic changes and the like are left out.
 */
import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import type { MemoryRecord } from '../records.js';
import { record } from './text.js';

/** Message subtypes left out by default: events about the channel, not things said in it. */
export const SLACK_SKIPPED_SUBTYPES = ['channel_join', 'channel_leave', 'channel_topic', 'channel_purpose', 'channel_name',
  'channel_archive', 'channel_unarchive', 'channel_convert_to_private', 'channel_convert_to_public', 'group_join', 'group_leave',
  'group_topic', 'group_purpose', 'group_name', 'group_archive', 'group_unarchive', 'pinned_item', 'unpinned_item', 'bot_add',
  'bot_remove', 'tombstone', 'reminder_add', 'joiner_notification', 'joiner_notification_for_inviter', 'message_deleted',
  'sh_room_created', 'channel_canvas_updated'];

export interface SlackOptions {
  /** Only these conversations (by name or id). */
  channels?: string[];
  /** Default: the channel's name. */
  collection?: string;
  /** Default: the export folder's name. */
  source?: string;
  /** Subtypes to leave out. Default: SLACK_SKIPPED_SUBTYPES. */
  skipSubtypes?: string[];
  /** Leave out messages posted by bots and integrations. */
  skipBots?: boolean;
  /** The workspace's address (`https://example.slack.com`), for a link to each message. */
  workspaceUrl?: string;
}

interface SlackUser { id: string; name?: string; real_name?: string; profile?: { real_name?: string; display_name?: string } }
interface SlackChannel { id: string; name?: string; members?: string[] }
interface SlackFile { title?: string; name?: string }
interface SlackAttachment { title?: string; fallback?: string; title_link?: string; from_url?: string }
interface SlackMessage {
  type?: string; subtype?: string; hidden?: boolean; user?: string; username?: string; bot_id?: string; text?: string; ts?: string;
  thread_ts?: string; reply_count?: number; user_profile?: { real_name?: string; display_name?: string; name?: string };
  bot_profile?: { name?: string }; files?: SlackFile[]; attachments?: SlackAttachment[];
}

async function readJson<T>(path: string): Promise<T | undefined> {
  try {
    return JSON.parse((await readFile(path, 'utf8')).replace(/^﻿/, '')) as T;
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw new Error(`cannot read ${path}: ${(e as Error).message}`);
  }
}

function personName(u: { real_name?: string; display_name?: string; name?: string; profile?: SlackUser['profile'] }): string | undefined {
  return [u.profile?.real_name, u.real_name, u.profile?.display_name, u.display_name, u.name].map((n) => n?.trim()).find(Boolean);
}

/** Slack's markup as plain text: mentions as names, channel links as `#name`, links as `label (url)`. */
export function slackText(text: string, users: Map<string, string>, channels: Map<string, string>): string {
  return text
    .replace(/<([^<>\n]*)>/g, (whole, inner: string) => {
      const bar = inner.indexOf('|');
      const target = bar >= 0 ? inner.slice(0, bar) : inner;
      const label = bar >= 0 ? inner.slice(bar + 1) : '';
      if (target.startsWith('@')) return `@${users.get(target.slice(1)) ?? (label.replace(/^@/, '') || target.slice(1))}`;
      if (target.startsWith('#')) return `#${label || channels.get(target.slice(1)) || target.slice(1)}`;
      if (target.startsWith('!')) {
        const special = target.slice(1);
        if (/^(here|channel|everyone)$/.test(special)) return `@${special}`;
        return label || `@${special.replace(/\^.*$/, '')}`;
      }
      if (!target) return whole;
      if (target.startsWith('mailto:')) return label || target.slice(7);
      return label && label !== target ? `${label} (${target})` : target;
    })
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

/** Records from an unzipped Slack export folder. */
export async function* fromSlackExport(dir: string, opts: SlackOptions = {}): AsyncIterable<MemoryRecord> {
  const root = resolve(dir);
  const source = opts.source ?? basename(root);
  const skip = new Set(opts.skipSubtypes ?? SLACK_SKIPPED_SUBTYPES);
  const users = new Map<string, string>();
  for (const u of (await readJson<SlackUser[]>(join(root, 'users.json'))) ?? []) {
    const n = personName(u);
    if (u?.id && n) users.set(u.id, n);
  }
  // Each conversation: its id, its name, and the folder its days are in (a direct message's folder is its id).
  const conversations: Array<{ id: string; name: string; folder: string }> = [];
  for (const [file, byId] of [['channels.json', false], ['groups.json', false], ['mpims.json', false], ['dms.json', true]] as const) {
    for (const c of (await readJson<SlackChannel[]>(join(root, file))) ?? []) {
      if (!c?.id) continue;
      const name = byId ? (c.members ?? []).map((m) => users.get(m) ?? m).join(', ') || c.id : c.name || c.id;
      conversations.push({ id: c.id, name, folder: byId ? c.id : c.name || c.id });
    }
  }
  if (!conversations.length) {
    // No lists: every folder is a conversation named after it.
    for (const e of (await readdir(root, { withFileTypes: true })).filter((e) => e.isDirectory() && !e.name.startsWith('.'))) {
      conversations.push({ id: e.name, name: e.name, folder: e.name });
    }
  }
  const channelNames = new Map(conversations.map((c) => [c.id, c.name]));
  const wanted = opts.channels ? new Set(opts.channels.map((c) => c.replace(/^#/, ''))) : undefined;

  for (const c of conversations) {
    if (wanted && !wanted.has(c.name) && !wanted.has(c.id)) continue;
    // The folder's name comes from the export's own lists: `../x`, `a/b` or an absolute path must not lead out of it.
    const folder = resolve(root, c.folder);
    if (!c.folder || dirname(folder) !== root) continue;
    if (!(await stat(folder).catch(() => undefined))?.isDirectory()) continue;
    const days = (await readdir(folder)).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort();
    for (const day of days) {
      const messages = (await readJson<SlackMessage[]>(join(folder, day))) ?? [];
      if (!Array.isArray(messages)) continue;
      for (const m of messages) {
        if (!m || typeof m !== 'object' || !m.ts || (m.type && m.type !== 'message') || m.hidden) continue;
        if (m.subtype && skip.has(m.subtype)) continue;
        const bot = m.subtype === 'bot_message' || (!!m.bot_id && !m.user);
        if (bot && opts.skipBots) continue;
        const extras = [
          ...(m.files ?? []).map((f) => f?.title || f?.name).filter((t): t is string => !!t?.trim()).map((t) => `File: ${t}`),
          ...(m.attachments ?? []).map((a) => {
            const t = a?.title || a?.fallback;
            if (!t?.trim()) return '';
            const link = a.title_link || a.from_url;
            return `Attachment: ${slackText(t, users, channelNames)}${link && !t.includes(link) ? ` (${link})` : ''}`;
          }).filter(Boolean),
        ];
        const said = slackText(m.text ?? '', users, channelNames).trim();
        const text = [said, ...extras].filter(Boolean).join('\n');
        if (!text) continue;
        const author = (m.user && users.get(m.user)) || (m.user_profile && personName(m.user_profile)) || m.bot_profile?.name ||
          m.username || m.user || undefined;
        const ms = Math.round(Number(m.ts) * 1000);
        yield record({
          id: `${c.id}:${m.ts}`, text, kind: 'message', source, author, collection: opts.collection ?? c.name,
          thread: `${c.name}:${m.thread_ts ?? m.ts}`, at: Number.isFinite(ms) ? ms : undefined,
          url: opts.workspaceUrl ? `${opts.workspaceUrl.replace(/\/+$/, '')}/archives/${c.id}/p${m.ts.replace('.', '')}` : undefined,
          metadata: { channel: c.name, ts: m.ts, threadTs: m.thread_ts, subtype: m.subtype, replies: m.reply_count },
        });
      }
    }
  }
}
