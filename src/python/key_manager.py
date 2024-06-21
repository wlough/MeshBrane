##########################################
# Key managment
class KeyCounter:
    """
    Key counter for generating unique key that increments with each call

    Properties
    ----------
    count : int
        integer-valued key

    Methods
    -------
    __call__(self) -> int
        return key and increment count
    """

    def __init__(self, start=0):
        self._count = start

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

    def __call__(self):
        key = self._count
        self._count += 1
        return key

    def increment(self):
        self._count += 1
        return self._count

    def decrement(self):
        self._count -= 1
        return self._count


class KeyManager:
    """
    Key manager which assigns a KeyCounter to each new key_type, or returns the existing KeyCounter for that key_type if it already exists.

    Properties
    ----------
    key_counter_dict : dict of (key_type, KeyCounter) key-value pairs
        {key_type0: KeyCounter0, key_type1: KeyCounter1,...}

    Methods
    -------
    __call__(self, key_type) -> int
        return and increment integer-valued key for key_type

    __getitem__(self, key_type) -> KeyCounter
        return KeyCounter instance for key_type
    """

    def __init__(self, key_types=[]):
        self._key_counter_dict = {key_type: KeyCounter() for key_type in key_types}

    def __call__(self, key_type):
        try:
            return self._key_counter_dict[key_type]()
        except KeyError:
            self._key_counter_dict[key_type] = KeyCounter()
            return self._key_counter_dict[key_type]()

    def __getitem__(self, key_type):
        try:
            return self._key_counter_dict[key_type]
        except KeyError:
            self._key_counter_dict[key_type] = KeyCounter()
            return self._key_counter_dict[key_type]
