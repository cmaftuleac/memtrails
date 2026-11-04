/**
 * The secret filter (redact.ts): what it must take out, what ordinary prose it must leave, and that no hostile string
 * makes it slow (it runs on every prompt and every model answer kept).
 */
import { describe, expect, it } from 'vitest';
import { redactString } from './redact.js';

const gone = (input: string, secret: string): void => {
  const out = redactString(input);
  expect(out, input).not.toContain(secret);
  expect(out, input).toContain('<redacted');
};

describe('redactString', () => {
  it('takes the value of a secret-named variable, whatever its prefix', () => {
    gone('AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY', 'wJalrXUtnFEMI');
    gone('export GITHUB_TOKEN=abcdef1234567890abcdef', 'abcdef1234567890');
    gone('DB_PASSWORD=hunter2hunter', 'hunter2');
    gone('PGPASSWORD=s3cretpw psql -h db', 's3cretpw');
    gone('access_token: abc123def456ghi', 'abc123def456');
    gone('client_secret=zzzyyyxxx111', 'zzzyyyxxx111');
    gone('AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net', 'abc+def');
    gone('Endpoint=sb://x.servicebus.windows.net/;SharedAccessKey=Zm9vYmFyYmF6', 'Zm9vYmFyYmF6');
    gone('password = hunter2', 'hunter2');
    expect(redactString('AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net')).toContain(';EndpointSuffix=core.windows.net');
  });

  it('takes JSON-style secrets, spaces and escaped quotes included', () => {
    expect(redactString('{"api_key": "abcd1234efgh5678"}')).toBe('{"api_key": "<redacted>"}');
    expect(redactString('{"password":"hunter2 with space"}')).toBe('{"password":"<redacted>"}');
    gone(`{'client_secret': 'q\\'uote secret'}`, 'uote');
    gone('{"refresh_token":"1//0gAbCdEf"}', '0gAbCdEf');
  });

  // Fake keys put together here, so a secret scanner reading this file sees no key.
  const fake = (...parts: string[]): string => parts.join('');
  it('takes provider key formats', () => {
    gone(fake('ci uses glpat', '-abcdefghij1234567890 for pushes'), 'glpat-');
    gone(fake('HF hf', '_abcdefghijklmnopqrstuvwxyzABCD'), 'hf_abc');
    gone(fake('npm', '_abcdefghijklmnopqrstuvwxyz0123456789'), 'npm_abc');
    gone('temporary ASIAABCDEFGHIJKLMNOP', 'ASIAABCDEFGHIJKLMNOP');
    gone('AKIAABCDEFGHIJKLMNOP', 'AKIAABCDEFGHIJKLMNOP');
    gone(fake('SG', '.abcdefghijklmnopqrstuv', '.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ'), 'abcdefghijklmnopqrstuv');
    gone(fake('bot 123456789', ':ABCdefGHIjklMNOpqrSTUvwxYZ012345678 ok'), 'ABCdefGHI');
    gone('sk-abcdefghijklmnop1234', 'sk-abc');
    gone('ghp_abcdefghijklmnopqrstuvwxyz', 'ghp_');
  });

  it('takes the credential of an Authorization header, scheme or not', () => {
    expect(redactString('Authorization: token ghx1234567890abcdef')).toBe('Authorization: token <redacted>');
    gone('-H "Authorization: Bearer abc.def.ghijklmnop"', 'abc.def');
    gone('authorization=abcdefghijkl', 'abcdefghijkl');
  });

  it('takes the whole password from a URL, `@` and `/` in it included', () => {
    expect(redactString('postgres://user:p@ss@host/db')).toBe('postgres://user:<redacted>@host/db');
    expect(redactString('https://user:pa/ss@host/x')).toBe('https://user:<redacted>@host/x');
    expect(redactString('see https://u:pw@example.org and ok')).toBe('see https://u:<redacted>@example.org and ok');
    expect(redactString('redis://:s3cret@cache:6379')).toBe('redis://:<redacted>@cache:6379');
  });

  it('still takes what it took before', () => {
    gone('-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----', 'MIIabc');
    gone('-----BEGIN PRIVATE KEY-----\nMIIunterminated', 'MIIunterminated');
    expect(redactString('mail ana.pop@example.org now')).toBe('mail <redacted:email>@example.org now');
    gone('https://x.org/cb?access_token=abcdef&x=1', 'abcdef');
    gone('eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U', 'eyJhbGci');
    expect(redactString('/Users/ana/projects/x')).toBe('~/projects/x');
    expect(redactString('a'.repeat(60))).toBe('<redacted:entropy>');
  });

  it('leaves ordinary prose, counts and commit shas alone', () => {
    for (const s of [
      'the password field is required',
      'token-based auth is simpler',
      'max_tokens: 4096',
      'token_count: 12',
      'fixed in 0123456789abcdef0123456789abcdef01234567 yesterday',
      'commit 9fceb02d0ae598e95dc970b74767f19372d61af8',
      'author: Jane Doe',
      'key: value',
      'a secret santa party',
      'http://localhost:8080/path',
      'meet at 10:30, then 12:45',
    ]) expect(redactString(s), s).toBe(s);
  });

  it('takes 40 hex characters when a secret name says what they are', () => {
    gone('GITHUB_TOKEN=0123456789abcdef0123456789abcdef01234567', '0123456789abcdef');
  });

  it('stays linear on hostile input (200k characters, each well under a second)', () => {
    const n = 200_000;
    const hostile = [
      'a.'.repeat(n / 2) + '@', 'x@' + 'a.'.repeat(n / 2), '-a'.repeat(n / 2) + '@', 'a@'.repeat(n / 2),
      'eyJ-'.repeat(n / 4), 'eyJaaaaaaaaa.'.repeat(n / 13), 'a://b:'.repeat(n / 6), 'a://b:@'.repeat(n / 7),
      'x://u:p@' + 'a'.repeat(n), '-----BEGIN PRIVATE KEY-----'.repeat(n / 27), 'password' + ' '.repeat(n),
      'a_'.repeat(n / 2) + '=', 'password='.repeat(n / 9), '"password": "'.repeat(n / 14), 'Bearer '.repeat(n / 7),
      '1234567890:'.repeat(n / 11), 'SG.' + 'a'.repeat(n), 'a:'.repeat(n / 2), '/Users/'.repeat(n / 7), ' '.repeat(n),
    ];
    for (const s of hostile) {
      const t = performance.now();
      redactString(s);
      const ms = performance.now() - t; if (process.env.SHOW) console.log(s.slice(0, 12), Math.round(ms));
      expect(ms, s.slice(0, 20)).toBeLessThan(1000);
    }
  });
});
