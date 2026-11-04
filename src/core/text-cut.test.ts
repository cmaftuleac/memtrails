/**
 * Cutting text that goes to Qdrant (a title, a key): never half a surrogate pair — Qdrant refuses the whole batch over
 * one. And the keyword channel's length cap applied to each part of an identifier, not to the whole of it.
 */
import { describe, expect, it } from 'vitest';
import { cut, tokens } from './text.js';
import { stmTitle } from './writer.js';

/** A UTF-16 unit of a surrogate pair standing alone: not valid UTF-8 once encoded. */
const LONE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

describe('cut', () => {
  it('never ends on half of an emoji', () => {
    const s = `${'a'.repeat(99)}😀 and more`;
    expect(LONE.test(s.slice(0, 100))).toBe(true); // what a plain slice did
    const c = cut(s, 100);
    expect(LONE.test(c)).toBe(false);
    expect(c).toBe(`${'a'.repeat(99)}…`);
    // A cut just after the pair keeps it whole.
    expect(cut(s, 101)).toBe(`${'a'.repeat(99)}😀…`);
  });

  it('leaves a short text as it is', () => {
    expect(cut('😀', 2)).toBe('😀');
    expect(cut('abc', 3)).toBe('abc');
  });

  it('gives a record a title Qdrant accepts when its first line is cut inside an emoji', () => {
    const title = stmTitle({ request: `${'x'.repeat(99)}🎉 party`, title: 'thread' });
    expect(LONE.test(title)).toBe(false);
  });
});

describe('keyword tokens of a long identifier', () => {
  it('splits a path longer than one keyword into its parts', () => {
    const path = 'services/infrastructure/certificates/renewal/hooks/letsencrypt_renewal_hook.sh';
    expect(path.length).toBeGreaterThan(60);
    const t = tokens(`run ${path}`);
    expect(t).not.toContain(path);
    expect(t).toEqual(expect.arrayContaining(['letsencrypt', 'renewal', 'hook', 'sh', 'certificates', 'services']));
  });

  it('still drops a part too long to be a word', () => {
    const blob = 'a'.repeat(80);
    expect(tokens(`x/${blob}/y`)).not.toContain(blob);
  });
});
