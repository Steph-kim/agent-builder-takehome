"""Time zones and date handling for extensions.

The model only ever emits local wall-clock times ("YYYY-MM-DDTHH:MM"); code attaches
the zone of the return location. The API sometimes emits datetimes in UTC (seen live
on AVS-77001020), so everything shown to a customer is converted to the location's
zone first, with the weekday spelled out so a wrong "Friday" is visible.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timezone, tzinfo
from zoneinfo import ZoneInfo

# US pilot airports. Unknown codes fall back to the reservation's own non-UTC offset.
IATA_TZ = {
    "LAX": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "OAK": "America/Los_Angeles",
    "SJC": "America/Los_Angeles",
    "SNA": "America/Los_Angeles",
    "SAN": "America/Los_Angeles",
    "BUR": "America/Los_Angeles",
    "SMF": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "PDX": "America/Los_Angeles",
    "LAS": "America/Los_Angeles",
    "PHX": "America/Phoenix",
    "DEN": "America/Denver",
    "SLC": "America/Denver",
    "DFW": "America/Chicago",
    "DAL": "America/Chicago",
    "IAH": "America/Chicago",
    "HOU": "America/Chicago",
    "AUS": "America/Chicago",
    "ORD": "America/Chicago",
    "MDW": "America/Chicago",
    "MSP": "America/Chicago",
    "ATL": "America/New_York",
    "MIA": "America/New_York",
    "FLL": "America/New_York",
    "MCO": "America/New_York",
    "TPA": "America/New_York",
    "CLT": "America/New_York",
    "DTW": "America/Detroit",
    "BOS": "America/New_York",
    "JFK": "America/New_York",
    "LGA": "America/New_York",
    "EWR": "America/New_York",
    "PHL": "America/New_York",
    "DCA": "America/New_York",
    "IAD": "America/New_York",
    "BWI": "America/New_York",
    "HNL": "Pacific/Honolulu",
    "ANC": "America/Anchorage",
}

_LOCAL_INPUT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")


class InvalidLocalTime(ValueError):
    """The model's local time is malformed or doesn't exist in that zone (DST gap)."""


def resolve_zone(location_code: str, reservation_datetimes: list[str]) -> tuple[tzinfo, str] | None:
    """Zone for a location → (tz, source). source is "iata" or "offset_fallback"; None = unresolvable."""
    name = IATA_TZ.get(location_code.upper())
    if name:
        return ZoneInfo(name), "iata"
    for raw in reservation_datetimes:
        offset = datetime.fromisoformat(raw).utcoffset()
        if offset:  # skip UTC-emitted values — they say nothing about the location
            return timezone(offset), "offset_fallback"
    return None


def parse_local(value: str, tz: tzinfo) -> datetime:
    """'2027-06-17T14:00' in `tz` → aware datetime. Rejects other formats and DST-gap times."""
    if not _LOCAL_INPUT.match(value.strip()):
        raise InvalidLocalTime(f"expected YYYY-MM-DDTHH:MM, got {value!r}")
    try:
        naive = datetime.fromisoformat(value.strip())
    except ValueError as e:
        raise InvalidLocalTime(str(e)) from None
    local = naive.replace(tzinfo=tz)
    if local.astimezone(UTC).astimezone(tz).replace(tzinfo=None) != naive:
        raise InvalidLocalTime(f"{value} does not exist in this time zone (clocks change)")
    return local


def to_local(iso: str, tz: tzinfo) -> datetime:
    return datetime.fromisoformat(iso).astimezone(tz)


def to_api(dt: datetime) -> str:
    """Aware datetime → the API's format, e.g. 2027-06-17T14:00:00-07:00."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.isoformat(timespec="seconds")


def format_for_customer(dt: datetime) -> str:
    """e.g. 'Thursday, June 17, 2027 at 2:00 PM PDT'."""
    hour = dt.strftime("%I").lstrip("0")
    zone = dt.tzname() or dt.strftime("%z")
    return f"{dt.strftime('%A, %B')} {dt.day}, {dt.year} at {hour}:{dt.strftime('%M %p')} {zone}"


def availability_dates(current_return: datetime, new_return: datetime, tz: tzinfo) -> tuple[str, str]:
    """Local calendar dates for /availability (the API takes dates, not datetimes)."""
    start: date = current_return.astimezone(tz).date()
    end: date = new_return.astimezone(tz).date()
    return start.isoformat(), end.isoformat()
