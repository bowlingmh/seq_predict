class LogStore:
    """Stores a "logarithmic number" of objects.  Keeps more recently added 
    objects.

    The class is also indexable, with newer objects first and the oldest 
    object last.
    
    >>> s = LogStore()
    >>> for i in range(16):
    ...     s.add(i)
    >>> list(s)
    [15, 14, 12, 8, 0]
    >>> s[-1]
    0
    >>> s[0]
    15
    """
    
    def __init__(self):
        self._items = []
        self._save = []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for x in self._items:
            yield x

    def __getitem__(self, i):
        return self._items[i]

    def lazy_add(self, x):
        self.add(x())
    
    def add(self, x):
        if not self._items:
            self._items.append(x)
            self._save.append(True)
        else:
            i = 0
            for i in range(len(self)):
                if self._save[i]:
                    self._items[i], x = x, self._items[i]
                    self._save[i] = False
                else:
                    self._items[i] = x
                    self._save[i] = True
                    return
            self._items.append(x)
            self._save.append(True)


class LogStoreUniform:
    """Stores a "logarithmic number" of objects.  Uniformly distributed.

    The class is also indexable, with newer objects first and the oldest 
    object last.
    
    >>> s = LogStoreUniform()
    >>> for i in range(16):
    ...     s.add(i)
    >>> list(s)
    [12, 8, 4, 0]
    >>> s[-1]
    0
    >>> s[0]
    12
    """
    
    def __init__(self):
        self.items = []
        self.gap = 1
        self.index_to_remove = -1
        self.num_to_skip = self.gap - 1

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for x in reversed(self.items):
            yield x

    def __getitem__(self, i):
        if i < 0: i += len(self.items)
        return self.items[len(self.items) - i - 1]
        
    def add(self, x):
        self.lazy_add(lambda: x)
        
    def lazy_add(self, x):
        if self.num_to_skip:
            self.num_to_skip -= 1
            return

        self.items.append(x())
        if self.index_to_remove > 0: del self.items[self.index_to_remove]
        self.index_to_remove += 1
        if self.index_to_remove >= len(self.items):
            self.gap *= 2
            self.index_to_remove = 0

        self.num_to_skip = self.gap - 1

        
