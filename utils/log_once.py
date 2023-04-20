_MESSAGE_KEYS = set()


def log_once(log_fn, key):
    if key not in _MESSAGE_KEYS:
        log_fn()
        _MESSAGE_KEYS.add(key)
