"""
What the reference library gets from JavaScript itself and Python does differently, reproduced here once so every
module reads like its TypeScript twin:

  - strings measured and cut in UTF-16 code units (``length``, ``slice``), as JavaScript counts them;
  - ``JSON.stringify``: the logs, the checkpoints and the fingerprints are byte for byte what Node writes;
  - ``Number.prototype.toString`` and ``String(n)``;
  - ``Date.parse``: V8's reader of ISO 8601 and of the legacy forms (RFC 2822 mail dates among them), local time
    included, so a record's time is the same number in both libraries;
  - ``toISOString`` and the few ``Intl`` date forms the prompts use.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from array import array
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

# --- strings in UTF-16 code units -------------------------------------------------------------------------------


def to16(s: str) -> str:
    """The string as UTF-16 code units, one Python character each (a character past U+FFFF becomes its surrogate
    pair): ``len``, slicing and indexing on it are JavaScript's ``length``, ``slice`` and ``[i]``."""
    if s.isascii():
        return s
    units = array("H", s.encode("utf-16-le", "surrogatepass"))
    if sys.byteorder == "big":
        units.byteswap()
    return "".join(map(chr, units))


def from16(u: str) -> str:
    """Code units back to a string; a surrogate pair is one character again, a lone surrogate stays as it is (as a
    JavaScript string can hold one)."""
    if u.isascii():
        return u
    return u.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def u16len(s: str) -> int:
    """``s.length`` in JavaScript."""
    if s.isascii():
        return len(s)
    return len(s) + sum(1 for c in s if ord(c) > 0xFFFF)


def u16slice(s: str, start: int, end: int | None = None) -> str:
    """``s.slice(start, end)`` in JavaScript: negative positions count from the end, in code units."""
    if s.isascii():
        return s[start:end]
    return from16(to16(s)[start:end])


def cut(s: str, n: int) -> str:
    """The first ``n`` UTF-16 units and an ellipsis — never ending on half a surrogate pair (an emoji cut in two is not
    valid UTF-8, and Qdrant refuses the whole write over it)."""
    u = to16(s)
    if len(u) <= n:
        return s
    high = n > 0 and 0xD800 <= ord(u[n - 1]) <= 0xDBFF
    end = n - 1 if high else n
    return f"{from16(u[:end])}…"


# JavaScript's \s: ECMAScript WhiteSpace and LineTerminator.
JS_SPACE = "\t\n\v\f\r    -     　﻿"
JS_WS = f"[{JS_SPACE}]"

_WS_RUN = re.compile(f"{JS_WS}+")


def flatten_ws(s: str) -> str:
    """``s.replace(/\\s+/g, ' ')``."""
    return _WS_RUN.sub(" ", s)


def js_trim(s: str) -> str:
    """``s.trim()``: JavaScript's whitespace, not Python's (which also takes U+001C–U+001F and U+0085)."""
    return s.strip("\t\n\v\f\r                  　﻿")


def js_trim_end(s: str) -> str:
    return s.rstrip("\t\n\v\f\r                  　﻿")


def js_is_space(c: str) -> bool:
    """``/\\s/.test(c)`` for one character."""
    return c in "\t\n\v\f\r       　﻿" or " " <= c <= " "


# --- numbers and JSON ---------------------------------------------------------------------------------------------


#: Every whole number up to this is exact in a double (Number.MAX_SAFE_INTEGER).
_SAFE_INT = 2**53 - 1


def num_str(x: float | int) -> str:
    """``String(x)`` for a number, as JavaScript writes it."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        if abs(x) <= _SAFE_INT:
            return str(x)
        # Past 2^53 JavaScript holds the nearest double, and writes its shortest digits: 12345678901234567890 is
        # "12345678901234567000".
        try:
            x = float(x)
        except OverflowError:
            return "Infinity" if x > 0 else "-Infinity"
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    if x.is_integer() and abs(x) <= _SAFE_INT:
        return str(int(x))
    _, digits_t, exp = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, digits_t)).rstrip("0") or "0"
    k = len(digits)
    n = int(exp) + len(digits_t)  # the decimal point's place: value = 0.digits × 10^n
    neg = "-" if x < 0 else ""
    if k <= n <= 21:
        return neg + digits + "0" * (n - k)
    if 0 < n <= 21:
        return neg + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return neg + "0." + "0" * (-n) + digits
    e = n - 1
    mant = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{neg}{mant}e{'+' if e >= 0 else '-'}{abs(e)}"


_ARRAY_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
_LONE = re.compile("[\ud800-\udfff]")


def _key_order(keys: Sequence[str]) -> list[str]:
    """An object's keys in JavaScript's order: array indices first, ascending, then the rest as inserted."""
    idx = [k for k in keys if _ARRAY_INDEX.match(k) and int(k) < 4294967295]
    if not idx:
        return list(keys)
    rest = [k for k in keys if k not in set(idx)]
    return sorted(idx, key=int) + rest


def _str(s: str) -> str:
    out = json.encoder.encode_basestring(s)
    if _LONE.search(out):
        # A lone surrogate is written as an escape (well-formed JSON.stringify); a pair stays as its character.
        out = _LONE.sub(lambda m: f"\\u{ord(m.group(0)):04x}", out)
    return str(out)


def stringify(v: Any) -> str:
    """``JSON.stringify(v)``: no spaces, keys in JavaScript's order, numbers as JavaScript writes them; ``None`` is
    ``null`` and a key whose value is ``UNDEFINED`` is left out."""
    parts: list[str] = []
    _write(v, parts)
    return "".join(parts)


class _Undefined:
    """JavaScript's ``undefined`` inside an object: the key is not written."""

    def __repr__(self) -> str:
        return "UNDEFINED"


UNDEFINED: Any = _Undefined()


def _write(v: Any, out: list[str]) -> None:
    if v is None or v is UNDEFINED:
        out.append("null")
    elif v is True:
        out.append("true")
    elif v is False:
        out.append("false")
    elif isinstance(v, str):
        out.append(_str(v))
    elif isinstance(v, int):
        out.append(str(v))
    elif isinstance(v, float):
        out.append(num_str(v) if math.isfinite(v) else "null")
    elif isinstance(v, Mapping):
        out.append("{")
        first = True
        for k in _key_order([str(k) for k in v]):
            x = v[k]
            if x is UNDEFINED:
                continue
            if not first:
                out.append(",")
            first = False
            out.append(_str(k))
            out.append(":")
            _write(x, out)
        out.append("}")
    elif isinstance(v, (list, tuple)):
        out.append("[")
        for i, x in enumerate(v):
            if i:
                out.append(",")
            _write(x, out)
        out.append("]")
    else:
        try:
            import numpy as np

            if isinstance(v, np.floating):
                _write(float(v), out)
                return
            if isinstance(v, np.integer):
                _write(int(v), out)
                return
        except ImportError:  # pragma: no cover - numpy is a dependency
            pass
        raise TypeError(f"cannot write {type(v).__name__} as JSON")


def utf8(s: str) -> bytes:
    """``Buffer.from(s)``: UTF-8, a lone surrogate written as U+FFFD."""
    return _LONE.sub("\ufffd", s).encode("utf-8")


def parse(text: str | bytes) -> Any:
    """``JSON.parse``: a whole number comes back an ``int`` (so it is written back without ``.0``)."""
    return json.loads(text, parse_constant=_no_constant)


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def intish(x: float) -> int | float:
    """A whole number as an ``int``, so it prints as JavaScript prints it."""
    return int(x) if math.isfinite(x) and float(x).is_integer() and abs(x) < 2**63 else x


# --- dates --------------------------------------------------------------------------------------------------------

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_WEEKDAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
_MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_LONG = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def utc_of(ms: float) -> datetime:
    return _EPOCH + timedelta(milliseconds=ms)


def iso(ms: float) -> str:
    """``new Date(ms).toISOString()``."""
    d = utc_of(math.floor(ms))
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{d.hour:02d}:{d.minute:02d}:{d.second:02d}.{d.microsecond // 1000:03d}Z"


def iso_day(ms: float) -> str:
    """``new Date(ms).toISOString().slice(0, 10)``."""
    return iso(ms)[:10]


def weekday_short(ms: float) -> str:
    """``toLocaleDateString('en-GB', { weekday: 'short', timeZone: 'UTC' })``."""
    return _WEEKDAY[(utc_of(ms).weekday() + 1) % 7]


def day_month_year(ms: float) -> str:
    """``toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' })``."""
    d = utc_of(ms)
    return f"{d.day} {_MONTH_LONG[d.month - 1]} {d.year}"


def date_utc(y: int, m: int, d: int = 1, h: int = 0, mi: int = 0, s: int = 0, ms: int = 0) -> int:
    """``Date.UTC(y, m, d, …)``: the month counted from 0 and allowed to run over, a year 0–99 read as 1900–1999, as
    JavaScript reads them."""
    if 0 <= y <= 99:
        y += 1900
    return int(_make_day(y, m, d) * 86_400_000 + h * 3_600_000 + mi * 60_000 + s * 1000 + ms)


def utc_weekday(ms: float) -> int:
    """``new Date(ms).getUTCDay()``: 0 for Sunday."""
    return (utc_of(ms).weekday() + 1) % 7


def utc_parts(ms: float) -> tuple[int, int, int]:
    """Year, month (from 0) and day of ``ms`` in UTC."""
    d = utc_of(ms)
    return d.year, d.month - 1, d.day


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= m <= 2
    era = int((y if y >= 0 else y - 399) / 400)  # truncating, as the algorithm's C division is
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _make_day(year: float, month: float, day: float) -> float:
    ym = year + math.floor(month / 12)
    mn = int(month % 12)
    return _days_from_civil(int(ym), mn + 1, 1) + day - 1


def _local_offset_ms(utc_ms: float) -> int:
    try:
        return int(time.localtime(utc_ms / 1000).tm_gmtoff) * 1000
    except (OverflowError, OSError, ValueError):
        return 0


def _local_to_utc(local_ms: float) -> float:
    guess = local_ms - _local_offset_ms(local_ms)
    off = _local_offset_ms(guess)
    return local_ms - off


_MAX_TIME = 8.64e15
_NONE = 2147483647  # V8's kNone: a slot not filled

_MONTH_NAME, _AM_PM, _TZ_NAME, _TIME_SEP, _INVALID = 1, 2, 3, 4, 0
_KEYWORDS: list[tuple[str, int, int]] = [
    ("jan", _MONTH_NAME, 1),
    ("feb", _MONTH_NAME, 2),
    ("mar", _MONTH_NAME, 3),
    ("apr", _MONTH_NAME, 4),
    ("may", _MONTH_NAME, 5),
    ("jun", _MONTH_NAME, 6),
    ("jul", _MONTH_NAME, 7),
    ("aug", _MONTH_NAME, 8),
    ("sep", _MONTH_NAME, 9),
    ("oct", _MONTH_NAME, 10),
    ("nov", _MONTH_NAME, 11),
    ("dec", _MONTH_NAME, 12),
    ("am\0", _AM_PM, 0),
    ("pm\0", _AM_PM, 12),
    ("ut\0", _TZ_NAME, 0),
    ("utc", _TZ_NAME, 0),
    ("z\0\0", _TZ_NAME, 0),
    ("gmt", _TZ_NAME, 0),
    ("cdt", _TZ_NAME, -5),
    ("cst", _TZ_NAME, -6),
    ("edt", _TZ_NAME, -4),
    ("est", _TZ_NAME, -5),
    ("mdt", _TZ_NAME, -6),
    ("mst", _TZ_NAME, -7),
    ("pdt", _TZ_NAME, -7),
    ("pst", _TZ_NAME, -8),
    ("t\0\0", _TIME_SEP, 0),
]


def _v8_space(c: str) -> bool:
    return js_is_space(c)


class _Tok:
    __slots__ = ("kind", "kw", "length", "value")

    def __init__(self, kind: str, value: int = 0, length: int = 0, kw: int = -1) -> None:
        self.kind = kind  # num, sym, ws, kw, unknown, end, invalid
        self.value = value
        self.length = length
        self.kw = kw

    def is_num(self) -> bool:
        return self.kind == "num"

    def is_fixed(self, n: int) -> bool:
        return self.kind == "num" and self.length == n

    def is_sym(self, c: str) -> bool:
        return self.kind == "sym" and self.value == ord(c)

    def is_sign(self) -> bool:
        return self.kind == "sym" and self.value in (43, 45)

    def sign(self) -> int:
        return 44 - self.value

    def is_kw(self, t: int | None = None) -> bool:
        return self.kind == "kw" and (t is None or self.kw == t)

    def is_z(self) -> bool:
        return self.kind == "kw" and self.kw == _TZ_NAME and self.length == 1 and self.value == 0

    def is_end(self) -> bool:
        return self.kind == "end"


class _Scanner:
    """V8's DateStringTokenizer over the string's UTF-16 units."""

    def __init__(self, s: str) -> None:
        self.s = to16(s)
        self.i = 0
        self.next_tok = self._scan()

    def _ch(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else "\0"

    def _scan(self) -> _Tok:
        s = self.s
        start = self.i
        if self.i >= len(s):
            return _Tok("end")
        c = s[self.i]
        if "0" <= c <= "9":
            while self._ch() == "0":
                self.i += 1
            n = 0
            k = 0
            while self.i < len(s) and "0" <= s[self.i] <= "9":
                if k < 9:
                    n = n * 10 + ord(s[self.i]) - 48
                k += 1
                self.i += 1
            return _Tok("num", n, self.i - start)
        if c in ":-+.)":
            self.i += 1
            return _Tok("sym", ord(c))
        if c >= "A" and not _v8_space(c):
            pre = ["\0", "\0", "\0"]
            n = 0
            while self.i < len(s) and s[self.i] >= "A" and not _v8_space(s[self.i]):
                if n < 3:
                    pre[n] = chr(ord(s[self.i]) | 0x20) if ord(s[self.i]) < 0x10000 else s[self.i]
                n += 1
                self.i += 1
            word = "".join(pre)
            for name, kind, value in _KEYWORDS:
                if word == name and (n <= 3 or kind == _MONTH_NAME):
                    return _Tok("kw", value, n, kind)
            return _Tok("kw", 0, n, _INVALID)
        if _v8_space(c):
            self.i += 1
            return _Tok("ws", 0, 1)
        if c == "(":
            balance = 0
            while True:
                ch = self._ch()
                if ch == ")":
                    balance -= 1
                elif ch == "(":
                    balance += 1
                self.i += 1
                if not (balance > 0 and self.i < len(s)):
                    break
            return _Tok("unknown")
        self.i += 1
        return _Tok("unknown")

    def next(self) -> _Tok:
        t = self.next_tok
        self.next_tok = self._scan()
        return t

    def peek(self) -> _Tok:
        return self.next_tok

    def skip(self, c: str) -> bool:
        if self.next_tok.is_sym(c):
            self.next()
            return True
        return False


def _between(x: int, lo: int, hi: int) -> bool:
    return lo <= x <= hi


class _Day:
    def __init__(self) -> None:
        self.comp: list[int] = []
        self.named = _NONE
        self.iso = False

    def add(self, n: int) -> bool:
        if len(self.comp) == 3:
            return False
        self.comp.append(n)
        return True

    def write(self) -> tuple[int, int, int] | None:
        if not self.comp:
            return None
        c = self.comp + [1] * (3 - len(self.comp))
        if self.named == _NONE:
            if self.iso or not _between(c[0], 1, 31):
                year, month, day = c[0], c[1], c[2]
            else:
                month, day, year = c[0], c[1], c[2]
        else:
            month = self.named
            if not _between(c[0], 1, 31):
                year, day = c[0], c[1]
            else:
                day, year = c[0], c[1]
        if not self.iso:
            if _between(year, 0, 49):
                year += 2000
            elif _between(year, 50, 99):
                year += 1900
        if not _between(month, 1, 12) or not _between(day, 1, 31):
            return None
        return year, month - 1, day


class _Time:
    def __init__(self) -> None:
        self.comp: list[int] = []
        self.hour_offset = _NONE

    def empty(self) -> bool:
        return not self.comp

    def expecting(self, n: int) -> bool:
        i = len(self.comp)
        return (i == 1 and _between(n, 0, 59)) or (i == 2 and _between(n, 0, 59)) or (i == 3 and _between(n, 0, 999))

    def add(self, n: int) -> bool:
        if len(self.comp) >= 4:
            return False
        self.comp.append(n)
        return True

    def add_final(self, n: int) -> bool:
        if not self.add(n):
            return False
        self.comp += [0] * (4 - len(self.comp))
        return True

    def write(self) -> tuple[int, int, int, int] | None:
        c = self.comp + [0] * (4 - len(self.comp))
        hour, minute, second, ms = c
        if self.hour_offset != _NONE:
            if not _between(hour, 0, 12):
                return None
            hour = hour % 12 + self.hour_offset
        if not (_between(hour, 0, 23) and _between(minute, 0, 59) and _between(second, 0, 59) and _between(ms, 0, 999)):
            if hour != 24 or minute or second or ms:
                return None
        return hour, minute, second, ms


class _Zone:
    def __init__(self) -> None:
        self.sign = _NONE
        self.hour = _NONE
        self.minute = _NONE

    def set(self, hours: int) -> None:
        self.sign = -1 if hours < 0 else 1
        self.hour = abs(hours)
        self.minute = 0

    def expecting(self, n: int) -> bool:
        return self.hour != _NONE and self.minute == _NONE and _between(n, 0, 59)

    def is_utc(self) -> bool:
        return self.hour == 0 and self.minute == 0

    def empty(self) -> bool:
        return self.hour == _NONE

    def write(self) -> float | None:
        """The offset in seconds; None for local time."""
        if self.sign == _NONE:
            return None
        h = 0 if self.hour == _NONE else self.hour
        m = 0 if self.minute == _NONE else self.minute
        total = h * 3600 + m * 60
        return float(-total if self.sign < 0 else total)


def _read_ms(t: _Tok) -> int:
    n, length = t.value, t.length
    if length == 1:
        n *= 100
    elif length == 2:
        n *= 10
    elif length > 3:
        length = min(length, 9)
        factor = 1
        while length > 3:
            factor *= 10
            length -= 1
        n //= factor
    return n


def _es5(sc: _Scanner, day: _Day, tm: _Time, tz: _Zone) -> _Tok:
    if sc.peek().is_sign():
        sign_tok = sc.next()
        if not sc.peek().is_fixed(6):
            return sign_tok
        year = sc.next().value
        if sign_tok.sign() < 0 and year == 0:
            return sign_tok
        day.add(sign_tok.sign() * year)
    elif sc.peek().is_fixed(4):
        day.add(sc.next().value)
    else:
        return sc.next()
    if sc.skip("-"):
        if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 1, 12):
            return sc.next()
        day.add(sc.next().value)
        if sc.skip("-"):
            if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 1, 31):
                return sc.next()
            day.add(sc.next().value)
    if not sc.peek().is_kw(_TIME_SEP):
        if not sc.peek().is_end():
            return sc.next()
    else:
        sc.next()
        if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 0, 24):
            return _Tok("invalid")
        hour_is_24 = sc.peek().value == 24
        tm.add(sc.next().value)
        if not sc.skip(":"):
            return _Tok("invalid")
        if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 0, 59) or (hour_is_24 and sc.peek().value > 0):
            return _Tok("invalid")
        tm.add(sc.next().value)
        if sc.skip(":"):
            if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 0, 59) or (hour_is_24 and sc.peek().value > 0):
                return _Tok("invalid")
            tm.add(sc.next().value)
            if sc.skip("."):
                if not sc.peek().is_num() or (hour_is_24 and sc.peek().value > 0):
                    return _Tok("invalid")
                tm.add(_read_ms(sc.next()))
        if sc.peek().is_z():
            sc.next()
            tz.set(0)
        elif sc.peek().is_sym("+") or sc.peek().is_sym("-"):
            tz.sign = 1 if sc.next().value == 43 else -1
            if sc.peek().is_fixed(4):
                hm = sc.next().value
                h, m = hm // 100, hm % 100
                if not _between(h, 0, 23) or not _between(m, 0, 59):
                    return _Tok("invalid")
                tz.hour, tz.minute = h, m
            else:
                if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 0, 23):
                    return _Tok("invalid")
                tz.hour = sc.next().value
                if not sc.skip(":"):
                    return _Tok("invalid")
                if not sc.peek().is_fixed(2) or not _between(sc.peek().value, 0, 59):
                    return _Tok("invalid")
                tz.minute = sc.next().value
        if not sc.peek().is_end():
            return _Tok("invalid")
    if tz.empty() and tm.empty():
        tz.set(0)
    day.iso = True
    return _Tok("end")


def date_parse(s: str) -> float:
    """``Date.parse(s)`` as V8 reads it: NaN for what it cannot read."""
    sc = _Scanner(s)
    day, tm, tz = _Day(), _Time(), _Zone()
    token = _es5(sc, day, tm, tz)
    if token.kind == "invalid":
        return math.nan
    has_number = bool(day.comp)
    while not token.is_end():
        if token.is_num():
            has_number = True
            n = token.value
            if sc.skip(":"):
                if sc.skip(":"):
                    if not tm.empty():
                        return math.nan
                    tm.add(n)
                    tm.add(0)
                else:
                    if not tm.add(n):
                        return math.nan
                    if sc.peek().is_sym("."):
                        sc.next()
            elif sc.skip(".") and tm.expecting(n):
                tm.add(n)
                if not sc.peek().is_num():
                    return math.nan
                ms = _read_ms(sc.next())
                if ms < 0:
                    return math.nan
                tm.add_final(ms)
            elif tz.expecting(n):
                tz.minute = n
            elif tm.expecting(n):
                tm.add_final(n)
                p = sc.peek()
                if not (p.is_end() or p.kind == "ws" or p.is_z() or p.is_sign()):
                    return math.nan
            else:
                if not day.add(n):
                    return math.nan
                sc.skip("-")
        elif token.is_kw():
            if token.kw == _AM_PM and not tm.empty():
                tm.hour_offset = token.value
            elif token.kw == _MONTH_NAME:
                day.named = token.value
                sc.skip("-")
            elif token.kw == _TZ_NAME and has_number:
                tz.set(token.value)
            else:
                if has_number:
                    return math.nan
                if sc.peek().is_num():
                    return math.nan
        elif token.is_sign() and (tz.is_utc() or not tm.empty()):
            tz.sign = token.sign()
            n = 0
            length = 0
            if sc.peek().is_num():
                t = sc.next()
                length, n = t.length, t.value
            has_number = True
            if sc.peek().is_sym(":"):
                tz.hour = n
                tz.minute = _NONE
            elif length in (1, 2):
                tz.hour, tz.minute = n, 0
            elif length in (3, 4):
                tz.hour, tz.minute = n // 100, n % 100
            else:
                return math.nan
        elif (token.is_sign() or token.is_sym(")")) and has_number:
            return math.nan
        token = sc.next()
    d = day.write()
    hms = tm.write()
    if d is None or hms is None:
        return math.nan
    off = tz.write()
    date = _make_day(*d) * 86_400_000 + hms[0] * 3_600_000 + hms[1] * 60_000 + hms[2] * 1000 + hms[3]
    if off is None:
        if abs(date) > _MAX_TIME + 864_000_000:
            return math.nan
        date = _local_to_utc(date)
    else:
        date -= off * 1000
    if not math.isfinite(date) or abs(date) > _MAX_TIME:
        return math.nan
    return float(date)


def local_ms(dt: datetime) -> float:
    """A ``datetime`` as milliseconds: an aware one by its zone, a naive one as local time (as a JavaScript Date)."""
    if dt.tzinfo is None:
        return _local_to_utc(
            (_days_from_civil(dt.year, dt.month, dt.day) * 86_400_000)
            + dt.hour * 3_600_000
            + dt.minute * 60_000
            + dt.second * 1000
            + dt.microsecond // 1000
        )
    return math.floor((dt - _EPOCH).total_seconds() * 1000)


def _collation_key(s: str) -> tuple[tuple[tuple[int, str], ...], tuple[str, ...], tuple[int, ...]]:
    chars = [c for c in unicodedata.normalize("NFD", s)]
    primary: list[tuple[int, str]] = []
    secondary: list[str] = []
    tertiary: list[int] = []
    for c in chars:
        cat = unicodedata.category(c)
        if cat.startswith("M"):
            if secondary:
                secondary[-1] += c
            continue
        if cat.startswith("L"):
            primary.append((2, c.casefold()))
            tertiary.append(1 if c != c.lower() else 0)
        elif cat == "Nd":
            primary.append((1, str(unicodedata.digit(c))))
            tertiary.append(0)
        else:
            primary.append((0, c))
            tertiary.append(0)
        secondary.append("")
    return tuple(primary), tuple(secondary), tuple(tertiary)


def locale_compare(a: str, b: str) -> int:
    """``a.localeCompare(b)`` as ICU's root collation orders the strings memory compares (names, ids): spaces and
    punctuation before digits before letters, letters by their base first (case and accents ignored), then accents
    (none first), then case (lower first). Strings that differ only where ICU ranks finer are ordered by code point."""
    ka, kb = _collation_key(a), _collation_key(b)
    if ka != kb:
        return -1 if ka < kb else 1
    return (a > b) - (a < b)
