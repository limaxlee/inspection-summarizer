"""Period arithmetic. Everything is naive UTC, matching how pymongo returns dates with tz_aware=False."""
from datetime import datetime, timedelta, timezone

from inspection_statistics.common.constants import Granularity

PERIOD_LENGTH: dict[Granularity, timedelta] = {
    Granularity.HOURLY: timedelta(hours=1),
    Granularity.SHIFT: timedelta(hours=12),
    Granularity.DAILY: timedelta(days=1),
    Granularity.WEEKLY: timedelta(days=7),
}


def to_naive_utc(value: datetime) -> datetime:
    """Aware datetimes are converted to UTC and stripped; naive ones are taken as UTC already."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def period_start(value: datetime, granularity: Granularity) -> datetime:
    """The start of the period that contains ``value``.

    2026-09-21 05:36 (a Monday) ->  hourly 05:00 | shift 00:00 | daily 00:00 | weekly Mon 2026-09-21 00:00
    """
    value = to_naive_utc(value)
    day = value.replace(hour=0, minute=0, second=0, microsecond=0)

    if granularity == Granularity.HOURLY:
        return value.replace(minute=0, second=0, microsecond=0)
    if granularity == Granularity.SHIFT:
        return day.replace(hour=12) if value.hour >= 12 else day
    if granularity == Granularity.DAILY:
        return day
    if granularity == Granularity.WEEKLY:
        return day - timedelta(days=day.weekday())
    raise ValueError(f"Unknown granularity: {granularity!r}")


def period_end(start: datetime, granularity: Granularity) -> datetime:
    return start + PERIOD_LENGTH[granularity]


def snap_window(start: datetime, end: datetime, granularity: Granularity) -> tuple[datetime, datetime]:
    """Widen [start, end) outward so that it covers only complete periods.

    A statistics document is a full recomputation of its period, so a window
    that cuts a period in half would produce a document missing half its data.
    An ``end`` that already sits on a boundary is left alone.
    """
    start, end = to_naive_utc(start), to_naive_utc(end)
    snapped_start = period_start(start, granularity)
    snapped_end = period_end(period_start(end - timedelta(microseconds=1), granularity), granularity)
    return snapped_start, snapped_end


def iter_periods(start: datetime, end: datetime, granularity: Granularity):
    """Yield every complete [period_start, period_end) covering [start, end)."""
    if end <= start:
        return

    cursor, snapped_end = snap_window(start, end, granularity)
    length = PERIOD_LENGTH[granularity]
    while cursor < snapped_end:
        yield cursor, cursor + length
        cursor += length
