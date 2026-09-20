from datetime import datetime, timedelta


def truncate_to_day_start(value: datetime) -> datetime:
    """Zero out the time part, keeping the date. tzinfo is preserved as-is."""
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def iter_day_ranges(start: datetime, end: datetime):
    """Yield half-open [day_start, day_end) windows clipped to [start, end).

    Walking the range day by day keeps one day of output in memory at a time and
    makes each day a self-contained, retryable unit of work.
    """
    if end <= start:
        return

    cursor = truncate_to_day_start(start)
    while cursor < end:
        next_day = cursor + timedelta(days=1)
        yield max(cursor, start), min(next_day, end)
        cursor = next_day
