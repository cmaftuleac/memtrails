"""
Profiles — what memory knows about one subject, gathered in one place (sleep's last step).
"""

from __future__ import annotations

PROFILE_MIN_ITEMS = 3
PROFILE_BROAD = 0.2
#: The items a profile reads: the newest this many, oldest first, each line cut to PROFILE_LINE.
PROFILE_READ = 40
#: Profiles a night writes (a large first sleep goes on the next nights), and asks in flight.
PROFILES_PER_SLEEP = 400
#: What memory_search and gathering show of a profile: all of it, up to this (measured p90 1,082 characters).
PROFILE_SHOWN = 1500
