from model import Model
from cffi import FFI
import re

class CTW_KT(Model):
    """Context Tree Weighting over KT models with a binary alphabet.

    A specialized memory and time efficient version of model.CTW with
    its default arguments.
    """

    ctwnode_h = open("fast_ctwnode.h", "r").read()
    ctwnode_c = open("fast_ctwnode.c", "r").read()
    _lib_cache = {}
    
    __slots__ = [ "tree", "depth", "c" ]

    @classmethod
    def make_lib(cls, alphabet_size):
        if alphabet_size not in cls._lib_cache:
            ffi = FFI()
            ffi.cdef(re.sub("ALPHABET_SIZE", str(alphabet_size), cls.ctwnode_h))
            cls._lib_cache[alphabet_size] = ffi.verify(re.sub("ALPHABET_SIZE", str(alphabet_size), cls.ctwnode_c))
        return cls._lib_cache[alphabet_size]

    def __init__(self, depth, alphabet_size = 2):
        self.c = self.make_lib(alphabet_size)
        self.depth = depth
        self.tree = self.c.ctwnode_new()
        
    def __del__(self):
        self.c.ctwnode_free(self.tree)

    def _mkcontext(self, x):
        padding = self.depth - len(x)
        return bytes([0] * padding + x[-self.depth:])
        
    def update(self, symbol, history):
        context = self._mkcontext(history)
        return self.c.ctwnode_update(self.tree, bytes([symbol]), context, self.depth)

    def log_predict(self, symbol, history):
        context = self._mkcontext(history)
        return self.c.ctwnode_log_predict(self.tree, bytes([symbol]), context, self.depth)

    @property
    def size(self):
        return self.c.ctwnode_size(self.tree)

    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.c = self.c
        r.depth = self.depth
        r.tree = self.c.ctwnode_copy(self.tree)
        return r
        
