import datetime
import pytz


def get_allowed_threads() -> int:
    '''
    moscow time logic for thread allocation
    to do not restrict other cluster projects through prime-time
    mon-fri 08:00-19:00 -> 3 threads
    mon-fri 19:00-08:00 -> 6 threads
    fri 18:00 - mon 07:00 -> 9 threads
    '''
    #

    tz = pytz.timezone('Europe/Moscow')
    now = datetime.datetime.now(tz)

    weekday = now.weekday()  # 0 = mon, 4 = fri, 6 = sun
    hour = now.hour

    # weekend mode check
    is_weekend_mode = (
            (weekday == 4 and hour >= 18) or
            (weekday == 5) or
            (weekday == 6) or
            (weekday == 0 and hour < 7)
    )

    if is_weekend_mode:
        return 9

    # weekday logic
    if 8 <= hour < 19:
        return 3
    else:
        return 6