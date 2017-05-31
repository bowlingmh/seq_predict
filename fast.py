from cffi import FFI
import math
import re
import os

from .model import Model

class FFIMixin:
    """Class mixin to handle loading FFI code with different parameterizations.

    Derived classes must spcify two variables `_ffi_code_h` and `_ffi_code_c`.

    You can retrieve a library module by calling:
    `_ffi(PARAMETER1=value, PARAMETER2=value)`
    """
    _ffi_cache = {}

    @classmethod
    def _ffi_prep_code(cls, defs, code):
        for d,v in defs:
            code = re.sub(d, str(v), code)
        return code

    @classmethod
    def _ffi(cls, **kwargs):
        defs = tuple(sorted(kwargs.items()))
        if defs not in cls._ffi_cache:
            ffi = FFI()
            ffi.cdef(cls._ffi_prep_code(defs, cls._ffi_code_h))
            cls._ffi_cache[defs] = ffi.verify(cls._ffi_prep_code(defs, cls._ffi_code_c), extra_compile_args=["-std=c99"])
        return cls._ffi_cache[defs]

    
class CTW_KT(Model, FFIMixin):
    """Context Tree Weighting over KT models with a fixed integer alphabet.

    A specialized memory and time efficient version of model.CTW with a KT
    estimator as the base model.
    """

    _ffi_code_h = open(os.path.join(os.path.dirname(__file__), "fast_ctwnode.h"), "r").read()
    _ffi_code_c = open(os.path.join(os.path.dirname(__file__), "fast_ctwnode.c"), "r").read()
    
    __slots__ = [ "tree", "depth", "c" ]

    def __init__(self, depth, alphabet_size=2, kt_sum_counts=1.0,
                 mkcontext=None):
        self.mkcontext = mkcontext if mkcontext else self._mkcontext 
        self.c = self._ffi(ALPHABET_SIZE=alphabet_size,
                           KT_SUM_COUNTS=kt_sum_counts)
        self.depth = depth
        self.tree = self.c.ctwnode_new()
        
    def __del__(self):
        self.c.ctwnode_free(self.tree)

    def _mkcontext(self, x):
        padding = self.depth - len(x)
        return bytes([0] * padding + x[-self.depth:])
        
    def update(self, symbol, history):
        context = self.mkcontext(history)
        return self.c.ctwnode_update(self.tree, symbol, context, self.depth)

    def log_predict(self, symbol, history):
        context = self.mkcontext(history)
        return self.c.ctwnode_log_predict(self.tree, symbol, context, self.depth)

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
        

class CTS_KT(Model, FFIMixin):
    """Context Tree Weighting over KT models with a fixed integer alphabet.

    A specialized memory and time efficient version of model.CTS with a KT
    estimator as the base model.
    """

    _ffi_code_h = open(os.path.join(os.path.dirname(__file__), "fast_ctsnode.h"), "r").read()
    _ffi_code_c = open(os.path.join(os.path.dirname(__file__), "fast_ctsnode.c"), "r").read()
    
    __slots__ = [ "tree", "depth", "c", "t" ]

    def __init__(self, depth, alphabet_size=2,
                 kt_sum_counts=0.125, base_prior=0.125,
                 mkcontext=None):
        self.mkcontext = mkcontext if mkcontext else self._mkcontext 
        self.c = self._ffi(ALPHABET_SIZE=alphabet_size,
                           KT_SUM_COUNTS=kt_sum_counts,
                           BASE_PRIOR=base_prior)
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
        context = self.mkcontext(history)

        log_alpha = math.log(self.alpha)
        log_blend = math.log(1 - 2 * self.alpha)
        self.t += 1

        lp = self.c.ctsnode_update(self.tree, symbol, context, self.depth,
                                   log_alpha, log_blend)
        return lp

    def log_predict(self, symbol, history):
        context = self.mkcontext(history)
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
        
    
