import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { fromSlackExport } from './slack.js';

async function all<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe('fromSlackExport', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memtrails-slack-'));
  const json = (path: string, value: unknown): void => {
    mkdirSync(join(dir, path, '..'), { recursive: true });
    writeFileSync(join(dir, path), JSON.stringify(value, null, 2));
  };
  json('users.json', [
    { id: 'U1', name: 'ana', real_name: 'Ana Popescu', profile: { real_name: 'Ana Popescu', display_name: 'ana' } },
    { id: 'U2', name: 'tanaka', profile: { real_name: '', display_name: '田中' } },
    { id: 'U3', name: 'omar' },
  ]);
  json('channels.json', [{ id: 'C1', name: 'general' }, { id: 'C2', name: 'proiecte' }]);
  json('general/2024-03-01.json', [
    { type: 'message', subtype: 'channel_join', user: 'U3', text: '<@U3> has joined the channel', ts: '1709290000.000100' },
    { type: 'message', user: 'U1', text: 'Planul: <@U2> preia <#C2|proiecte>, vezi <https://example.org/doc|documentul> și <https://example.org/raw>. 1 &lt; 2 &amp;&amp; ok',
      ts: '1709290800.000200', thread_ts: '1709290800.000200', reply_count: 1 },
    { type: 'message', user: 'U2', text: '了解しました <@U1|ana> 👍', ts: '1709291000.000300', thread_ts: '1709290800.000200' },
    { type: 'message', user: 'U3', text: '', ts: '1709291100.000400', files: [{ name: 'تقرير.pdf', title: 'التقرير الشهري' }] },
    { type: 'message', user: 'U3', text: '', ts: '1709291150.000450' },
    { type: 'message', subtype: 'bot_message', bot_id: 'B1', username: 'deploy-bot', text: 'Deployed <!here>',
      attachments: [{ title: 'Build 42', title_link: 'https://example.org/b/42' }], ts: '1709291200.000500' },
  ]);
  json('general/2024-03-02.json', [
    { type: 'message', user: 'U1', text: 'Mulțumesc! <#C1>', ts: '1709377200.000100' },
    { type: 'message', subtype: 'channel_topic', user: 'U1', text: 'set the topic', ts: '1709377300.000200' },
  ]);
  json('proiecte/2024-03-01.json', [{ type: 'message', user: 'U3', text: 'مرحبا بالجميع', ts: '1709290900.000100' }]);

  it('reads every message that says something, with names, links and threads', async () => {
    const records = await all(fromSlackExport(dir, { workspaceUrl: 'https://example.slack.com/' }));
    expect(records.map((r) => r.id)).toEqual(['C1:1709290800.000200', 'C1:1709291000.000300', 'C1:1709291100.000400',
      'C1:1709291200.000500', 'C1:1709377200.000100', 'C2:1709290900.000100']);
    const [root, reply, file, bot, later, other] = records;
    expect(root.text).toBe('Planul: @田中 preia #proiecte, vezi documentul (https://example.org/doc) și https://example.org/raw. 1 < 2 && ok');
    expect(root).toMatchObject({ author: 'Ana Popescu', kind: 'message', collection: 'general', thread: 'general:1709290800.000200',
      at: 1709290800000, url: 'https://example.slack.com/archives/C1/p1709290800000200' });
    expect(root.metadata).toMatchObject({ replies: 1 });
    expect(reply).toMatchObject({ text: '了解しました @Ana Popescu 👍', author: '田中', thread: 'general:1709290800.000200' });
    expect(file).toMatchObject({ text: 'File: التقرير الشهري', author: 'omar', thread: 'general:1709291100.000400' });
    expect(bot).toMatchObject({ text: 'Deployed @here\nAttachment: Build 42 (https://example.org/b/42)', author: 'deploy-bot' });
    expect(later.text).toBe('Mulțumesc! #general');
    expect(other).toMatchObject({ text: 'مرحبا بالجميع', collection: 'proiecte', source: records[0].source });
  });

  it('reads only the channels asked for, into the collection given, without bots when asked', async () => {
    const records = await all(fromSlackExport(dir, { channels: ['#general'], collection: 'team', skipBots: true }));
    expect(records).toHaveLength(4);
    expect(records.every((r) => r.collection === 'team' && r.id.startsWith('C1:'))).toBe(true);
  });
});
