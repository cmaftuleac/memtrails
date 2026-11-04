"""Records as memory keeps them (records.py; records.test.ts and records-refresh.test.ts, ported)."""

from __future__ import annotations

from datetime import datetime, timezone

from memtrails._js import date_parse
from memtrails.core.episodes import episode_body, render_for_extraction
from memtrails.records import collection_key, conversation_entries, entry_of, invalid_record, record_uid, time_of

NOW = int(date_parse("2026-05-01T00:00:00Z"))


def test_a_titled_record() -> None:
    e = entry_of(
        {
            "id": "m1",
            "source": "mailbox",
            "kind": "email",
            "title": "Proxy certificate",
            "author": "Ana",
            "text": "It expired on the 1st.\nThe renewal job had been removed.",
            "collection": "Ops team",
            "thread": "t-9",
            "at": "2026-04-02",
            "url": "mid:1",
            "metadata": {"to": "ops"},
        },
        NOW,
    )
    assert (e["engine"], e["session"], e["project"], e["title"], e["request"]) == (
        "email",
        "t-9",
        "Ops-team",
        "Proxy certificate",
        "Proxy certificate",
    )
    assert e["reply"] == "Ana: It expired on the 1st.\nThe renewal job had been removed."
    assert e["ts"] == date_parse("2026-04-02")
    assert e["files"] == ["mid:1"]
    assert e["record"] == {"id": "m1", "source": "mailbox", "author": "Ana", "url": "mid:1", "metadata": {"to": "ops"}}
    assert e["uid"] == record_uid({"id": "m1", "source": "mailbox"})
    assert episode_body(e) == "Proxy certificate\nAna: It expired on the 1st.\nThe renewal job had been removed."
    assert (
        render_for_extraction(e, 3)
        == "## Record 3 · 2026-04-02 00:00 · email\nProxy certificate\nAna: It expired on the 1st.\nThe renewal job had been removed."
    )


def test_the_first_line_is_the_title_in_any_script() -> None:
    e = entry_of({"id": "n", "text": "会议记录：部署推迟到周五\n原因是证书过期。"}, NOW)
    assert e["title"] == "会议记录：部署推迟到周五"
    assert e["reply"] == "原因是证书过期。"
    assert e["ts"] == NOW
    assert e["engine"] == "record"
    assert e["session"] == e["uid"]
    assert collection_key(None) == "default"


def test_the_fingerprint_changes_only_with_the_record() -> None:
    r = {"id": "x", "text": "a\nb", "author": "Ion"}
    assert entry_of(r, NOW)["fingerprint"] == entry_of(r, NOW + 1)["fingerprint"]
    assert entry_of({**r, "text": "a\nc"}, NOW)["fingerprint"] != entry_of(r, NOW)["fingerprint"]
    assert entry_of({**r, "metadata": {"k": 1}}, NOW)["fingerprint"] != entry_of(r, NOW)["fingerprint"]


def test_says_what_is_wrong() -> None:
    assert "id" in (invalid_record({"id": "", "text": "x"}) or "")
    assert "no text" in (invalid_record({"id": "a", "text": "  "}) or "")
    assert "time" in (invalid_record({"id": "a", "text": "x", "at": "not a date"}) or "")
    assert invalid_record({"id": "a", "text": "", "title": "only a title"}) is None
    assert time_of(datetime(1970, 1, 1, 0, 0, 0, 5000, tzinfo=timezone.utc)) == 5
    assert time_of("") is None


def test_dates_as_javascript_reads_them() -> None:
    assert time_of("Mon, 1 Jan 2024 10:00:00 +0100") == date_parse("2024-01-01T09:00:00Z")
    assert time_of("2024-03-01T10:00:00+02:00") == date_parse("2024-03-01T08:00:00Z")
    assert time_of("Tue, 02 Jan 2024 09:00:00 GMT") == 1704186000000
    assert time_of("yesterday-ish") is None
    assert time_of("2024-02-30") == date_parse("2024-03-01T00:00:00Z")  # V8 rolls a day past the month over
    assert time_of(float("nan")) is None


def test_a_conversation_is_cut_where_the_person_speaks() -> None:
    es = conversation_entries(
        {
            "id": "c1",
            "collection": "p",
            "at": NOW,
            "turns": [
                {"role": "assistant", "text": "Hello, how can I help?"},
                {"role": "user", "text": "Deploy failed."},
                {"role": "user", "text": "Logs say TLS."},
                {"role": "assistant", "text": "The certificate expired."},
                {"role": "assistant", "text": "Renew it with the cron job."},
                {"role": "user", "text": "Thanks", "author": "Ana"},
            ],
        },
        NOW,
    )
    assert [(e["request"], e["reply"]) for e in es] == [
        ("Hello, how can I help?", "Hello, how can I help?"),
        ("Deploy failed.\nLogs say TLS.", "The certificate expired.\nRenew it with the cron job."),
        ("Ana: Thanks", ""),
    ]
    assert [e["engine"] for e in es] == ["conversation"] * 3
    assert {e["session"] for e in es} == {"c1"}
    assert [e["turn"] for e in es] == [0, 1, 2]
    assert es[1]["said"] == es[1]["reply"]
    assert "Agent said:\nThe certificate expired." in render_for_extraction(es[1], 1)


TURNS = [
    {"role": "user", "text": "Deploy failed."},
    {"role": "assistant", "text": "The certificate expired."},
    {"role": "user", "text": "Renewed it."},
    {"role": "assistant", "text": "Good."},
]


def test_a_conversation_without_times_is_unchanged_when_added_again() -> None:
    first = conversation_entries({"id": "c", "turns": TURNS}, NOW)
    again = conversation_entries({"id": "c", "turns": TURNS}, NOW + 3_600_000)
    assert [e["fingerprint"] for e in again] == [e["fingerprint"] for e in first]
    assert [e["ts"] for e in first] == [NOW, NOW + 60_000]


def test_a_conversation_changes_only_where_a_time_changes() -> None:
    at = {"id": "c", "at": NOW, "turns": TURNS}
    a = conversation_entries(at, NOW)
    assert [e["fingerprint"] for e in conversation_entries(at, NOW + 1)] == [e["fingerprint"] for e in a]
    assert [e["ts"] for e in a] == [NOW, NOW + 60_000]
    assert conversation_entries({**at, "at": NOW + 5}, NOW)[0]["fingerprint"] != a[0]["fingerprint"]
    moved = conversation_entries({"id": "c", "turns": [TURNS[0], TURNS[1], {**TURNS[2], "at": NOW + 7}, TURNS[3]]}, NOW)
    plain = conversation_entries({"id": "c", "turns": TURNS}, NOW)
    assert moved[0]["fingerprint"] == plain[0]["fingerprint"]
    assert moved[1]["fingerprint"] != plain[1]["fingerprint"]
    assert moved[1]["ts"] == NOW + 7
