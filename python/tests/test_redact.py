"""The secret filter (redact.py): what it must take out, what ordinary prose it must leave, and that no hostile string
makes it slow — it runs on every prompt and every model answer kept. (redact.test.ts, ported; fake keys are put
together at run time, so a secret scanner reading this file sees no key.)"""

from __future__ import annotations

import time

import pytest

from memtrails.core.redact import redact_string


def gone(text: str, secret: str) -> None:
    out = redact_string(text)
    assert secret not in out, text
    assert "<redacted" in out, text


def fake(*parts: str) -> str:
    return "".join(parts)


def test_secret_named_variables() -> None:
    gone("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI")
    gone("export GITHUB_TOKEN=abcdef1234567890abcdef", "abcdef1234567890")
    gone("DB_PASSWORD=hunter2hunter", "hunter2")
    gone("PGPASSWORD=s3cretpw psql -h db", "s3cretpw")
    gone("access_token: abc123def456ghi", "abc123def456")
    gone("client_secret=zzzyyyxxx111", "zzzyyyxxx111")
    gone("AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net", "abc+def")
    gone("Endpoint=sb://x.servicebus.windows.net/;SharedAccessKey=Zm9vYmFyYmF6", "Zm9vYmFyYmF6")
    gone("password = hunter2", "hunter2")
    assert ";EndpointSuffix=core.windows.net" in redact_string("AccountKey=abc+def/ghi==;EndpointSuffix=core.windows.net")


def test_json_style_secrets() -> None:
    assert redact_string('{"api_key": "abcd1234efgh5678"}') == '{"api_key": "<redacted>"}'
    assert redact_string('{"password":"hunter2 with space"}') == '{"password":"<redacted>"}'
    gone("{'client_secret': 'q\\'uote secret'}", "uote")
    gone('{"refresh_token":"1//0gAbCdEf"}', "0gAbCdEf")


def test_provider_key_formats() -> None:
    gone(fake("ci uses glpat", "-abcdefghij1234567890 for pushes"), "glpat-")
    gone(fake("HF hf", "_abcdefghijklmnopqrstuvwxyzABCD"), "hf_abc")
    gone(fake("npm", "_abcdefghijklmnopqrstuvwxyz0123456789"), "npm_abc")
    gone("temporary ASIAABCDEFGHIJKLMNOP", "ASIAABCDEFGHIJKLMNOP")
    gone("AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP")
    gone(fake("SG", ".abcdefghijklmnopqrstuv", ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"), "abcdefghijklmnopqrstuv")
    gone(fake("bot 123456789", ":ABCdefGHIjklMNOpqrSTUvwxYZ012345678 ok"), "ABCdefGHI")
    gone(fake("sk-", "abcdefghijklmnop1234"), "sk-abc")
    gone(fake("ghp_", "abcdefghijklmnopqrstuvwxyz"), "ghp_")


def test_authorization_headers() -> None:
    assert redact_string("Authorization: token ghx1234567890abcdef") == "Authorization: token <redacted>"
    gone('-H "Authorization: Bearer abc.def.ghijklmnop"', "abc.def")
    gone("authorization=abcdefghijkl", "abcdefghijkl")


def test_the_whole_password_of_a_url() -> None:
    assert redact_string("postgres://user:p@ss@host/db") == "postgres://user:<redacted>@host/db"
    assert redact_string("https://user:pa/ss@host/x") == "https://user:<redacted>@host/x"
    assert redact_string("see https://u:pw@example.org and ok") == "see https://u:<redacted>@example.org and ok"
    assert redact_string("redis://:s3cret@cache:6379") == "redis://:<redacted>@cache:6379"


def test_the_rest() -> None:
    gone("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----", "MIIabc")
    gone("-----BEGIN PRIVATE KEY-----\nMIIunterminated", "MIIunterminated")
    assert redact_string("mail ana.pop@example.org now") == "mail <redacted:email>@example.org now"
    gone("https://x.org/cb?access_token=abcdef&x=1", "abcdef")
    gone("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "eyJhbGci")
    assert redact_string("/Users/ana/projects/x") == "~/projects/x"
    assert redact_string("a" * 60) == "<redacted:entropy>"
    assert redact_string(None) == ""
    assert redact_string(42) == ""


@pytest.mark.parametrize(
    "text",
    [
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
    ],
)
def test_leaves_prose_alone(text: str) -> None:
    assert redact_string(text) == text


def test_forty_hex_after_a_secret_name() -> None:
    gone("GITHUB_TOKEN=0123456789abcdef0123456789abcdef01234567", "0123456789abcdef")


N = 200_000
HOSTILE = [
    "a." * (N // 2) + "@",
    "x@" + "a." * (N // 2),
    "-a" * (N // 2) + "@",
    "a@" * (N // 2),
    "eyJ-" * (N // 4),
    "eyJaaaaaaaaa." * (N // 13),
    "a://b:" * (N // 6),
    "a://b:@" * (N // 7),
    "x://u:p@" + "a" * N,
    "-----BEGIN PRIVATE KEY-----" * (N // 27),
    "password" + " " * N,
    "a_" * (N // 2) + "=",
    "password=" * (N // 9),
    '"password": "' * (N // 14),
    "Bearer " * (N // 7),
    "1234567890:" * (N // 11),
    "SG." + "a" * N,
    "a:" * (N // 2),
    "/Users/" * (N // 7),
    " " * N,
    "-token" * (N // 6),
    "key." * (N // 4),
]


@pytest.mark.parametrize("text", HOSTILE, ids=lambda s: repr(s[:12]))
def test_stays_linear_on_hostile_input(text: str) -> None:
    t = time.perf_counter()
    redact_string(text)
    assert time.perf_counter() - t < 1.0
