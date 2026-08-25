from datetime import datetime, timedelta


def truncate_to_day_start(value: datetime) -> datetime:
    """Zero out the time part, keeping the date. tzinfo is preserved as-is."""
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def iter_day_ranges(start: datetime, end: datetime):
    """Yield half-open [day_start, day_end) windows clipped to [start, end).

    Summary documents are bucketed per UTC day, so walking the range day by day
    means every bucket touched inside one window is written exactly once.
    """
    if end <= start:
        return

    cursor = truncate_to_day_start(start)
    while cursor < end:
        next_day = cursor + timedelta(days=1)
        yield max(cursor, start), min(next_day, end)
        cursor = next_day
