/**
 * What memory indexes from text besides the content key: the keyword vector (BM25 term weights, IDF on Qdrant's
 * side) and the small-to-big children. Each case is a way a search would miss: an identifier split wrong, a term
 * weighted twice, a long body cut into one window or into windows that lose the end.
 */
import { describe, expect, it } from 'vitest';
import { children, conceptKey, docSparse, firstWords, querySparse, termId, tokens } from './text.js';

describe('keyword channel', () => {
  it('keeps identifiers whole and also by their parts, lower-cased', () => {
    const t = tokens('Run deploy.sh on billing_api; see apps/web/main.ts');
    expect(t).toEqual(expect.arrayContaining(['run', 'deploy.sh', 'deploy', 'sh', 'billing_api', 'billing', 'api', 'apps/web/main.ts', 'web', 'main', 'ts']));
    expect(t).not.toContain('');
  });

  it('keeps words in other scripts', () => {
    expect(tokens('Почему не работает выгрузка? Ce face funcția?')).toEqual(expect.arrayContaining(['почему', 'выгрузка', 'functia']));
  });

  it('matches a word typed with or without its accents', () => {
    expect(tokens('Bază de date în producție')).toEqual(tokens('baza de date in productie'));
    expect(tokens('Ёлка ПРОДАКШН')).toEqual(tokens('елка продакшн'));
  });

  it('finds the words of scripts written without spaces, and in mixed text', () => {
    expect(tokens('我们需要重建网关镜像然后重新创建容器')).toEqual(expect.arrayContaining(['重建', '容器']));
    expect(tokens('ゲートウェイのイメージを再構築する')).toEqual(expect.arrayContaining(['イメージ']));
    expect(tokens('สร้างอิมเมจเกตเวย์ใหม่')).toEqual(expect.arrayContaining(['เกตเวย์']));
    expect(tokens('请 rebuild 网关的 docker image')).toEqual(expect.arrayContaining(['rebuild', 'docker', 'image']));
  });

  it('keeps a word whose marks make it (Devanagari) whole', () => {
    expect(tokens('डेटाबेस को फिर से बनाओ')).toContain('डेटाबेस');
  });

  it('gives a term one id however often it appears, and weights repeats less than linearly (BM25)', () => {
    const v = docSparse('qdrant qdrant qdrant sidecar');
    expect(v.indices).toEqual([...v.indices].sort((a, b) => a - b));
    expect(new Set(v.indices).size).toBe(v.indices.length);
    const q = v.values[v.indices.indexOf(termId('qdrant'))]!;
    const s = v.values[v.indices.indexOf(termId('sidecar'))]!;
    expect(q).toBeGreaterThan(s);
    expect(q).toBeLessThan(3 * s);
  });

  it('asks each query term once, with weight 1', () => {
    const q = querySparse(['memory index', 'the memory']);
    expect(q.indices.length).toBe(new Set(q.indices).size);
    expect(q.values.every((x) => x === 1)).toBe(true);
    expect(q.indices).toContain(termId('memory'));
  });
});

describe('first words', () => {
  it('cuts by words in any script, keeping the text as written', () => {
    expect(firstWords('one  two\nthree four', 3)).toBe('one two three');
    expect(firstWords('我们需要重建网关镜像', 2)).toBe('我们需要');
  });
});

describe('children', () => {
  it('keeps a short body as one child', () => {
    expect(children('one two three')).toEqual(['one two three']);
  });

  it('cuts a long body into ~96-word windows that overlap a quarter and reach its end', () => {
    const words = Array.from({ length: 300 }, (_, i) => `w${i}`);
    const kids = children(words.join(' '));
    expect(kids.length).toBeGreaterThan(2);
    for (const k of kids) expect(k.split(' ').length).toBeLessThanOrEqual(96);
    expect(kids[1]!.startsWith('w72 ')).toBe(true);
    expect(kids.at(-1)).toContain('w299');
  });

  it('windows a body written without spaces', () => {
    const body = '我们需要重建网关镜像然后重新创建容器。'.repeat(40);
    const kids = children(body);
    expect(kids.length).toBeGreaterThan(1);
    expect(kids.every((k) => k.length < body.length)).toBe(true);
  });

  it('covers the whole of a long text: every word in a chunk, the last one included', () => {
    const words = Array.from({ length: 5000 }, (_, i) => `w${i}`);
    const kids = children(words.join(' '));
    const seen = new Set(kids.flatMap((k) => k.split(' ')));
    expect(words.every((w) => seen.has(w))).toBe(true);
    expect(kids.length).toBeGreaterThan(60);
  });
});

describe('concept keys', () => {
  it('are one node however a concept is typed: case, accents, spacing, stray punctuation — identifiers kept', () => {
    expect(conceptKey(' Proxy.conf ')).toBe('proxy.conf');
    expect(conceptKey('"Bază   de date",')).toBe('baza de date');
    expect(conceptKey('acme-proxy:dev')).toBe('acme-proxy:dev');
    expect(conceptKey('网关 镜像')).toBe('网关 镜像');
    expect(conceptKey('...')).toBe('');
  });
});
