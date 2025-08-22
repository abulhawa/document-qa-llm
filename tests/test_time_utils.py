from datetime import datetime, timezone

from utils.time_utils import format_date, format_timestamp


def test_format_date_handles_common_formats():
    dt = datetime(2004, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    epoch = dt.timestamp()

    # datetime instance
    assert format_date(dt) == "02 January 2004"
    # ISO string
    assert format_date(dt.isoformat()) == "02 January 2004"
    # ISO string with trailing Z
    assert format_date(dt.isoformat().replace("+00:00", "Z")) == "02 January 2004"
    # epoch as number and string
    assert format_date(epoch) == "02 January 2004"
    assert format_date(str(int(epoch))) == "02 January 2004"


def test_format_timestamp_handles_common_formats():
    dt = datetime(2023, 6, 7, 8, 9, 10, tzinfo=timezone.utc)
    epoch = dt.timestamp()

    assert format_timestamp(dt) == "2023-06-07 08:09:10"
    assert format_timestamp(dt.isoformat()) == "2023-06-07 08:09:10"
    assert format_timestamp(dt.isoformat().replace("+00:00", "Z")) == "2023-06-07 08:09:10"
    assert format_timestamp(epoch) == "2023-06-07 08:09:10"
    assert format_timestamp(str(int(epoch))) == "2023-06-07 08:09:10"
