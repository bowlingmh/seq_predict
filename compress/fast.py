from cffi import FFI
import math
import re
import os

from .model import Model

class CTW_KT(Model):
    """Context Tree Weighting over KT models with a fixed integer alphabet.

    A specialized memory and time efficient version of model.CTW with
    its default arguments.
    """

    ctwnode_h = open(os.path.join(os.path.dirname(__file__), "fast_ctwnode.h"), "r").read()
    ctwnode_c = open(os.path.join(os.path.dirname(__file__), "fast_ctwnode.c"), "r").read()
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
        

class CTS_KT(Model):
    """Context Tree Weighting over KT models with a fixed integer alphabet.

    A specialized memory and time efficient version of model.CTS with
    its default arguments.
    """

    ctsnode_h = open(os.path.join(os.path.dirname(__file__), "fast_ctsnode.h"), "r").read()
    ctsnode_c = open(os.path.join(os.path.dirname(__file__), "fast_ctsnode.c"), "r").read()
    _lib_cache = {}
    
    __slots__ = [ "tree", "depth", "c", "t" ]

    @classmethod
    def _prep_code(cls, defs, code):
        for d,v in defs:
            code = re.sub(d, str(v), code)
        return code
    
    @classmethod
    def _lib(cls, **kwargs):
        defs = tuple(sorted(kwargs.items()))
        if defs not in cls._lib_cache:
            ffi = FFI()
            ffi.cdef(cls._prep_code(defs, cls.ctsnode_h))
            cls._lib_cache[defs] = ffi.verify(cls._prep_code(defs, cls.ctsnode_c))
        return cls._lib_cache[defs]

    def __init__(self, depth, alphabet_size = 2):
        self.c = self._lib(ALPHABET_SIZE=2, KT_SUM_COUNTS=0.125, BASE_PRIOR=0.075)
        self.depth = depth
        self.tree = self.c.ctsnode_new()
        self.t = 0
        
    def __del__(self):
        self.c.ctsnode_free(self.tree)

    def _mkcontext(self, x):
        padding = self.depth - len(x)
        return bytes([0] * padding + x[-self.depth:])

    @property
    def alpha(self):
        return 1.0 / (self.t + 3)
    
    def update(self, symbol, history):
        context = self._mkcontext(history)

        log_alpha = math.log(self.alpha)
        log_blend = math.log(1 - 2 * self.alpha)
        self.t += 1

        lp = self.c.ctsnode_update(self.tree, symbol, context, self.depth,
                                   log_alpha, log_blend)
        #print(symbol, lp, math.log(1-math.exp(lp)))
        return lp

    def log_predict(self, symbol, history):
        context = self._mkcontext(history)
        return self.c.ctsnode_log_predict(self.tree, symbol, context, self.depth)

    @property
    def size(self):
        return self.c.ctsnode_size(self.tree)

    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.c = self.c
        r.depth = self.depth
        r.tree = self.c.ctsnode_copy(self.tree)
        r.t = self.t
        return r
        
    
