"""Arithmetic Coding

Functions for doing compression using arithmetic coding.
http://en.wikipedia.org/wiki/Arithmetic_coding

The functions and classes all need predictive models; see model.py
"""

import math
import itertools

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def compress_bits(model, bits):
    """Compresses a stream of bits into another stream of bits.
    Requires a prediction model.
    """
    encoder = BinaryArithmeticEncoder(model)

    for c in itertools.chain.from_iterable((encoder.encode(b) for b in bits)):
        yield c
    for c in encoder.flush():
        yield c

def compress_bytes(model, bytes):
    """Compresses a stream of bytes into another steam of bytes.
    Requires a prediction model.
    """
    bits = ((m >> i) & 1 for m in bytes for i in range(8))
    cbits = compress_bits(model, bits)
    for c in (int(''.join(byte), 2) for byte in grouper(8, (str(b) for b in cbits), '0')):
        yield c

def decompress_bits(model, bits, msglen):
    """Decompresses a stream of bits into another stream of bits.
    Requires the same prediction model (from its original state) that was
    used for decompression and the number of bits in the message.
    """
    decoder = BinaryArithmeticDecoder(model)
    nbits = 0
    for r in itertools.chain(*(decoder.decode(b) for b in bits)):
        yield r
        nbits += 1
    for r in decoder.flush(msglen - nbits):
        yield r

def decompress_bytes(model, bytes, msglen):
    """Decompresses a stream of bytes into another stream of bytes.
    Requires the same prediction model (from its original state) that was
    used for decompression and the number of bytes in the message.
    """
    cbits = ((m >> i) & 1 for m in bytes for i in range(8))
    bits = decompress_bits(model, cbits, msglen * 8)
    for r in (int(''.join(byte), 2) for byte in grouper(8, (str(b) for b in bits), '0')):
        yield r

class BinaryArithmeticEncoder:
    """BinaryArithmeticEncoder

    An arithmetic encoder for binary data sources.  For the theory behind the encoder
    see http://en.wikipedia.org/wiki/Arithmetic_coding.

    >>> encoder = BinaryArithmeticEncoder(CTW(8))

    See also: BinaryArithmeticDecoder, compress, and compress_bytes
    """
    
    def __init__(self, model, num_bits = 32):
        self.model = model
        self.num_bits = num_bits

        self._top = 2 ** self.num_bits 
        self._half = self._top // 2  # [0, self._half) is outputs the zero bit
        self._1_4 = self._half // 2
        self._3_4 = self._top - self._half

        self.low = 0 # Interval is [self.low, self.high)
        self.high = self._top 
        self.follow_bits = 0 # Opposing bits to follow the next output'd bit

        self.history = []

    def encode(self, symbol):
        """Encodes a symbol returning a sequence of coded bits.

        The encoder is stateful and (since it is hopefully compressing the input) it will not
        return output bits for each input symbol.

        You will need to flush the encoder to get remaining coded bits after encoding the 
        complete sequence.
        """
        
        output = []

        # Find the split point 
        p_symbol = math.exp(self.model.update(symbol, self.history))
        self.history.append(symbol)

        p_zero = p_symbol if symbol == 0 else 1 - p_symbol
        split = self.low + max(1, int((self.high - self.low) * p_zero)) # 0-interval is [self.low, split)

        # Update the range based on the observed symbol
        if symbol:
            self.low = split
        else:
            self.high = split

        # If the range no longer overlaps the midpoint, the next bit is known
        #   also rescale the interval to get back precision
        #
        # If the range overlaps the midpoint but not the 1/4 or 3/4 points then
        #   we rescale the interval, but track this with follow bits.  If the next
        #   bit to output is a 1, then we already know it's at the low end of the upper
        #   half, so we follow with a 0.  Similarly if the next bit is a 0, then
        #   we already know it's at the high end of the lower half, so we follow
        #   with a 1.
        # If this happens a second time before outputting any bit, then there will
        #   need to be 2 of these follow bits.  So we track this by just incrementing 
        #   a follow bit counter.
        #
        # This is in a loop because the new range may not overlap the new midpoint,
        #   allowing multiple bits to be determined
        output = []
        while True:
            if self.high <= self._half:
                output.append(0)
                output.extend([1] * self.follow_bits) # Add the follow bits
                self.follow_bits = 0
            elif self.low >= self._half:
                output.append(1)
                output.extend([0] * self.follow_bits) # Add the follow bits
                self.follow_bits = 0
                self.low -= self._half
                self.high -= self._half
            elif self.low >= self._1_4 and self.high <= self._3_4:
                self.follow_bits += 1
                self.low -= self._1_4
                self.high -= self._1_4
            else:
                break

            self.low *= 2
            self.high *= 2

        return output

    def flush(self):
        """Flushes any coded bits in the encoder.  Typically called after the entire
        sequence has been encoded.
        """
        if self.low < self._1_4:
            output = [0] + [1] * (self.follow_bits + 1)
        else:
            output = [1] + [0] * (self.follow_bits + 1)

        return output

class BinaryArithmeticDecoder:
    def __init__(self, model, num_bits = 32):
        self.model = model
        self.num_bits = num_bits
        self._top = 2 ** self.num_bits 
        self._half = self._top // 2 # [0, self._half) outputs the zero bit
        self._1_4 = self._half // 2
        self._3_4 = self._top - self._1_4
        self.low = 0 
        self.high = 1 # This ensures num_bits are read before decoding
        self.value = 0

    def decode(self, bit):
        if self.low >= self._half:
            self.value -= self._half
            self.low -= self._half
            self.high -= self._half
        elif self.low >= self._1_4 and self.high <= self._3_4:
            self.value -= self._1_4
            self.low -= self._1_4
            self.high -= self._1_4            

        self.low *= 2
        self.high *= 2
        self.value *= 2
        self.value += bit

        output = []
        
        while self.low < self._half < self.high:
            p_zero = self.model.predict(0)
            split = self.low + int((self.high - self.low) * p_zero) # 0-interval is [self.low, split)

            symbol = 0 if self.value < split else 1
            output.append(symbol)
            self.model.update(symbol)
            
            if symbol:
                self.low = split
            else:
                self.high = split

        return output

    def flush(self, nbits):
        output = []
        while len(output) < nbits:
            output += self.decode(0)
        return output[:nbits]
