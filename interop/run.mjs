#!/usr/bin/env node
// A memory folder written by one language is opened and searched by another with the same results.
//
//   node interop/run.mjs
//
// INTEROP_LANGS      which drivers take part (default ts,py,rs)
// INTEROP_PYTHON     the Python that has memtrails installed (default python/.venv/bin/python when it exists, else python3)
// INTEROP_WORK       where the homes go (default a new temporary folder, removed when everything agrees)
import { spawnSync } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const corpus = join(here, 'corpus.json');
const queries = join(here, 'queries.json');
// Embeddings come out a little different between languages on x64 processors (ONNX Runtime's 4-bit kernels): a
// score moved by up to 0.007 was seen on GitHub's Linux runners, on Apple silicon they agree to 1e-6. Within this,
// two hits whose scores tie may also trade places; anything further apart must be in the same order.
const SCORE_TOLERANCE = 1e-2;
const ci = process.env.GITHUB_ACTIONS === 'true';

const say = (line) => process.stdout.write(`${line}\n`);
const group = (title) => say(ci ? `::group::${title}` : `\n== ${title}`);
const endGroup = () => ci && say('::endgroup::');

const python = process.env.INTEROP_PYTHON
  ?? (existsSync(join(root, 'python/.venv/bin/python')) ? join(root, 'python/.venv/bin/python') : 'python3');
const rsBinary = join(here, 'rs/target/debug', process.platform === 'win32' ? 'memtrails-interop.exe' : 'memtrails-interop');
const drivers = {
  ts: [join(root, 'node_modules/.bin/tsx'), join(here, 'ts.ts')],
  py: [python, join(here, 'py.py')],
  rs: [rsBinary],
};

const langs = (process.env.INTEROP_LANGS ?? 'ts,py,rs').split(',').map((s) => s.trim()).filter(Boolean);
for (const l of langs) if (!drivers[l]) throw new Error(`unknown language ${l} (ts, py, rs)`);

// The Rust driver builds against ../../rust with the library's own lock file, so it gets the versions the library is
// tested with.
if (langs.includes('rs')) {
  group('Building the Rust driver');
  copyFileSync(join(root, 'rust/Cargo.lock'), join(here, 'rs/Cargo.lock'));
  const built = spawnSync('cargo', ['build'], { cwd: join(here, 'rs'), stdio: 'inherit' });
  endGroup();
  if (built.status !== 0) throw new Error(`the Rust driver does not build: ${built.error?.message ?? `cargo build exited with ${built.status}`}`);
}
if (langs.length < 2) throw new Error(`interop needs two languages at least, got ${langs.join(', ') || 'none'}`);

const work = process.env.INTEROP_WORK ? resolve(process.env.INTEROP_WORK) : mkdtempSync(join(tmpdir(), 'memtrails-interop-'));
mkdirSync(work, { recursive: true });
say(`languages: ${langs.join(', ')}; homes in ${work}`);

function run(lang, cmd, home, file) {
  const [bin, ...args] = drivers[lang];
  const started = Date.now();
  const r = spawnSync(bin, [...args, cmd, home, file], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 << 20 });
  const took = ((Date.now() - started) / 1000).toFixed(1);
  if (r.status !== 0) {
    say(r.stderr ?? '');
    throw new Error(`${lang} ${cmd} ${home} failed (${r.error?.message ?? `exit ${r.status}`})`);
  }
  say(`  ${lang} ${cmd} ${home.slice(work.length + 1)} (${took} s)`);
  return r.stdout;
}

function read(lang, home, label) {
  const out = run(lang, 'read', home, queries);
  writeFileSync(join(work, `${label}.json`), out);
  return normalize(JSON.parse(out));
}

/** One thing is not compared: a `match` that is the record's whole key again. A short record's only chunk holds the
 *  same text as its key, so the two points tie exactly and Qdrant returns either one — from run to run, in every
 *  language, the reference included. When the chunk wins, search reports it as the passage that matched (and
 *  searchText repeats the summary after " … "). A match that is a different passage is still compared. */
function normalize(read) {
  const hit = (h) => (h.match === h.summary ? { ...h, match: null } : h);
  for (const r of Object.values(read.results)) {
    for (const key of ['hits', 'found', 'related']) if (Array.isArray(r[key])) r[key] = r[key].map(hit);
    if (typeof r.text === 'string') {
      r.text = r.text.split('\n').map((line) => {
        if (!line.startsWith('   ')) return line;
        for (let at = line.indexOf(' … '); at >= 0; at = line.indexOf(' … ', at + 1)) {
          if (line.slice(3, at) === line.slice(at + 3)) return line.slice(0, at);
        }
        return line;
      }).join('\n');
    }
  }
  return read;
}

/** A copy of a home, so that every reader starts from the same files (a read writes usage.json and recall.jsonl). */
function copyHome(from, to) {
  rmSync(to, { recursive: true, force: true });
  cpSync(from, to, { recursive: true });
  return to;
}

// --- comparing ---------------------------------------------------------------------------------------------------

const show = (v) => (typeof v === 'number' ? String(Math.round(v * 1000) / 1000) : JSON.stringify(v));

const isHit = (h) => h && typeof h === 'object' && typeof h.score === 'number' && typeof h.id === 'string';

/** Hits in `got` put in `want`'s order where the two that trade places score within SCORE_TOLERANCE of each other. */
function nearTiesAsWanted(want, got) {
  if (!want.every(isHit) || !got.every(isHit)) return got;
  const out = [...got];
  for (let i = 0; i < Math.min(want.length, out.length); i++) {
    if (out[i].id === want[i].id) continue;
    const j = out.findIndex((h, k) => k > i && h.id === want[i].id);
    if (j > i && Math.abs(out[j].score - out[i].score) <= SCORE_TOLERANCE) [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function differences(want, got, path, out) {
  if (typeof want === 'number' && typeof got === 'number') {
    const ok = path.endsWith('.score') ? Math.abs(want - got) <= SCORE_TOLERANCE : want === got;
    if (!ok) out.push(`${path}: expected ${show(want)}, got ${show(got)}`);
    return out;
  }
  if (Array.isArray(want) && Array.isArray(got)) {
    got = nearTiesAsWanted(want, got);
    if (want.length !== got.length) out.push(`${path}: expected ${want.length} entries, got ${got.length}`);
    for (let i = 0; i < Math.min(want.length, got.length); i++) differences(want[i], got[i], `${path}[${i}]`, out);
    return out;
  }
  if (want && got && typeof want === 'object' && typeof got === 'object' && !Array.isArray(want) && !Array.isArray(got)) {
    for (const k of new Set([...Object.keys(want), ...Object.keys(got)])) {
      if (!(k in got)) out.push(`${path}.${k}: expected ${show(want[k])}, missing`);
      else if (!(k in want)) out.push(`${path}.${k}: not expected, got ${show(got[k])}`);
      else differences(want[k], got[k], `${path}.${k}`, out);
    }
    return out;
  }
  if (want !== got) out.push(`${path}: expected ${show(want)}, got ${show(got)}`);
  return out;
}

/** What a list of hits was, in one line: ids in order, with kind and score. */
const hitLine = (hits) => hits.map((h) => `${h.record ?? h.id} (${h.kind}, ${show(h.score)})`).join(', ');

const checks = [];
function compare(title, want, got) {
  const lines = [];
  for (const name of new Set([...Object.keys(want.results), ...Object.keys(got.results)])) {
    const d = differences(want.results[name], got.results[name], name, []);
    if (!d.length) continue;
    lines.push(`  query "${name}":`);
    for (const l of d.slice(0, 12)) lines.push(`    ${l}`);
    if (d.length > 12) lines.push(`    … and ${d.length - 12} more`);
    for (const key of ['hits', 'found']) {
      if (want.results[name]?.[key] && got.results[name]?.[key]) {
        lines.push(`    expected: ${hitLine(want.results[name][key])}`);
        lines.push(`    got:      ${hitLine(got.results[name][key])}`);
      }
    }
  }
  for (const l of differences(want.stats, got.stats, 'stats', [])) lines.push(`  ${l}`);
  checks.push({ title, ok: !lines.length });
  say(`${lines.length ? 'DIFFERS' : 'same   '} ${title}`);
  for (const l of lines) say(l);
}

// --- the runs ----------------------------------------------------------------------------------------------------

group('Writing a home in each language');
const homes = {};
for (const w of langs) {
  homes[w] = join(work, `home-${w}`);
  rmSync(homes[w], { recursive: true, force: true });
  run(w, 'write', homes[w], corpus);
}
endGroup();

group('Reading every home in every language');
const reads = {};
for (const w of langs) {
  reads[w] = {};
  for (const r of langs) reads[w][r] = read(r, copyHome(homes[w], join(work, `read-${w}-by-${r}`)), `read-${w}-by-${r}`);
}
endGroup();

group('Adding a record in each language to every home, and reading it in the language that wrote it');
const added = {};
for (const w of langs) {
  added[w] = {};
  for (const r of langs) {
    const home = copyHome(homes[w], join(work, `add-${w}-then-${r}`));
    run(r, 'add', home, corpus);
    added[w][r] = read(w, home, `add-${w}-then-${r}-read-by-${w}`);
  }
}
endGroup();

say('\nResults (scores within 1e-2, near ties in either order; everything else exactly):');
for (const w of langs) {
  for (const r of langs) if (r !== w) compare(`written by ${w}, read by ${r} — against ${w}'s own read`, reads[w][w], reads[w][r]);
}
for (const w of langs.slice(1)) compare(`written and read by ${w} — against ${langs[0]} written and read by ${langs[0]}`, reads[langs[0]][langs[0]], reads[w][w]);
for (const w of langs) {
  for (const r of langs) if (r !== w) compare(`written by ${w}, one record added by ${r}, read by ${w} — against ${w} adding it`, added[w][w], added[w][r]);
}

const failed = checks.filter((c) => !c.ok);
say(`\n${checks.length - failed.length} of ${checks.length} comparisons agree.`);
if (failed.length) {
  say(`Every read is kept in ${work} (read-<writer>-by-<reader>.json).`);
  for (const c of failed) say(ci ? `::error::interop: ${c.title}` : `FAILED: ${c.title}`);
  process.exit(1);
}
if (!process.env.INTEROP_WORK) rmSync(work, { recursive: true, force: true });
