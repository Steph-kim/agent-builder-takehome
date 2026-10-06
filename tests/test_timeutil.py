import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from avis_agent.timeutil import (
    InvalidLocalTime,
    availability_dates,
    format_for_customer,
    parse_local,
    resolve_zone,
    to_api,
    to_local,
)

FIXTURES = Path(__file__).parent / "fixtures"
LA = ZoneInfo("America/Los_Angeles")


def reservation(rid):
    return json.loads((FIXTURES / f"{rid}.json").read_text())


def dates_of(res):
    return list(res["dates"].values())


def test_known_airports_resolve_by_iata():
    tz, source = resolve_zone("DFW", [])
    assert source == "iata" and tz == ZoneInfo("America/Chicago")


def test_unknown_location_falls_back_to_reservation_offset_skipping_utc():
    tz, source = resolve_zone("XYZ", ["2026-10-05T09:50:34+00:00", "2026-06-27T08:00:00-07:00"])
    assert source == "offset_fallback" and tz.utcoffset(None) == timedelta(hours=-7)


def test_unknown_location_with_only_utc_datetimes_is_unresolvable():
    assert resolve_zone("XYZ", ["2026-10-05T09:50:34+00:00"]) is None


def test_parse_local_round_trips_to_api_format_with_dst_offset():
    assert to_api(parse_local("2027-06-17T14:00", LA)) == "2027-06-17T14:00:00-07:00"  # PDT
    assert to_api(parse_local("2027-01-17T14:00", LA)) == "2027-01-17T14:00:00-08:00"  # PST


@pytest.mark.parametrize(
    "bad",
    ["2027-06-17 14:00", "2027-06-17", "tomorrow at 2", "2027-13-01T10:00", "2027-06-17T14:00:00-07:00"],
)
def test_parse_local_rejects_anything_but_local_wall_clock(bad):
    with pytest.raises(InvalidLocalTime):
        parse_local(bad, LA)


def test_parse_local_rejects_times_skipped_by_spring_forward():
    with pytest.raises(InvalidLocalTime):
        parse_local("2027-03-14T02:30", LA)


def test_utc_emitted_return_is_shown_in_location_time():
    # Priya's return arrives from the API in UTC; the customer must see SFO local time.
    priya = reservation("AVS-77001020")
    tz, _ = resolve_zone(priya["return_location"]["code"], dates_of(priya))
    local = to_local(priya["dates"]["current_return_datetime"], tz)
    assert local.utcoffset() == timedelta(hours=-7)
    assert format_for_customer(local).startswith("Monday, October 5, 2026 at 2:50 AM PDT")


def test_customer_format_spells_out_weekday():
    shown = format_for_customer(parse_local("2027-06-17T14:00", LA))
    assert shown == "Thursday, June 17, 2027 at 2:00 PM PDT"


def test_availability_dates_are_local_calendar_days_not_utc():
    # 01:30 UTC on Oct 6 is still Oct 5 in San Francisco.
    current = to_local("2026-10-06T01:30:00+00:00", LA)
    new = parse_local("2026-10-07T18:00", LA)
    assert availability_dates(current, new, LA) == ("2026-10-05", "2026-10-07")
