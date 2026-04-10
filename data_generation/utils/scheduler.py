import datetime
import pytz


def get_allowed_threads() -> int:
    """
    Return the maximum number of concurrent LLM worker threads allowed at the
    current Moscow time.

    Thread limits are intentionally reduced during weekday business hours to
    avoid contention with other users on the shared inference cluster.

    Schedule:
      - Mon–Fri 08:00–19:00  →  3 threads  (business hours)
      - Mon–Fri 19:00–08:00  →  6 threads  (off-peak)
      - Fri 18:00 – Mon 07:00 →  9 threads  (weekend)
    """
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.datetime.now(tz)

    weekday = now.weekday()  # 0 = Monday, 4 = Friday, 6 = Sunday
    hour = now.hour

    # Weekend window: Friday evening through Monday early morning.
    is_weekend_mode = (
            (weekday == 4 and hour >= 18) or
            (weekday == 5) or
            (weekday == 6) or
            (weekday == 0 and hour < 7)
    )

    if is_weekend_mode:
        return 9

    # Standard weekday schedule: throttle during 08:00–19:00 MSK.
    if 8 <= hour < 19:
        return 3
    else:
        return 6