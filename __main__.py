"""General compression based on probabilistic models.

Usage: z.py compress INFILE OUTFILE [-m MODEL] [-d DEPTH]
       z.py decompress INFILE OUTFILE [-m MODEL] [-d DEPTH]

Options:
-m MODEL      Prediction model [default: CTW]
-d DEPTH      Depth parameter used for CTW [default: 48]
"""

from . import PTW, FMN, KT, CTW_KT, CTS_KT
from .ac import compress_bytes, decompress_bytes

import docopt
import os
import sys
import time

import resource

def progress_bar(frac):
    bar_length = 60
    prog_length = int(frac * bar_length) 

    bar = []
    bar = prog_length * '=' + ('>' + (bar_length - prog_length - 1) * ' ' if prog_length < bar_length else '')

    print('\r[{}]  {}%'.format(bar, int(frac * 100)), end='')

def _bytes_with_progress(file, nbytes, chunksize = 1024):
    count = 0
    while True:
        chunk = file.read(1024)
        if not chunk: return
        for b in chunk:
            count += 1
            yield b
        progress_bar(count / nbytes)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    depth = int(args['-d'])
    
    if args['-m'] == "CTW":
        probmodel = CTW_KT(depth)
    elif args['-m'] == "CTS":
        probmodel = CTS_KT(depth)
    elif args['-m'] == "PTW:CTW":
        probmodel = PTW(lambda: CTW_KT(depth))
    elif args['-m'] == "FMN:CTW":
        probmodel = FMN(lambda: CTW_KT(depth))
    elif args['-m'] == "KT":
        probmodel = KT()
    elif args['-m'] == "PTW":
        probmodel = PTW()
    elif args['-m'] == "FMN":
        probmodel = FMN()
    elif args['-m'] == "CTW:PTW":
        probmodel = CTW(depth, lambda: PTW())
    elif args['-m'] == "CTW:FMN":
        probmodel = CTW(depth, lambda: FMN())
    else:
        raise ValueError('Unknown model string')

    infile = args['INFILE']
    outfile = args['OUTFILE']
        
    if args['compress']:
        msglen = os.path.getsize(infile)
        codelen = 0

        print("Compressing {} ({} bytes)\n".format(infile, msglen))
        start_time = time.time()
        with open(infile, 'rb') as infs, open(outfile, 'wb') as outfs:
            outfs.write(msglen.to_bytes(4, sys.byteorder))
            
            for b in compress_bytes(probmodel, _bytes_with_progress(infs, msglen)):
                codelen += 1
                outfs.write(bytes([b]))
        elapsed_time = time.time() - start_time

        print("\n\nCompression statistics:")
        print("    {:15} {:7.4f}%".format("Ratio:", (codelen+4)/msglen * 100))
        print("    {:15} {:d} bytes".format("Size:", (codelen+4)))
        print("    {:15} {:7.5f}".format("Bits per Byte:", (codelen+4) * 8 / msglen))
        print("    {:15} {:7f}s".format("Time:", elapsed_time))
        print("    {:15} {:7f}".format("KB/s:", msglen / 1024 / elapsed_time))
        if hasattr(probmodel, 'size'): print("    {:15} {:d}".format("# of Nodes:", probmodel.size))
        print("    {:15} {:7.4f} MB".format("Memory Used:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))

    elif args['decompress']:
        msglen = os.path.getsize(infile)
        codelen = 0

        print("Decompressing {} ({} bytes)\n".format(infile, msglen))
        with open(infile, 'rb') as infs, open(outfile, 'wb') as outfs:
            codelen = os.path.getsize(infile)
            msglen = int.from_bytes(infs.read(4), sys.byteorder)

            for b in decompress_bytes(probmodel, _bytes_with_progress(infs, msglen), codelen):
                outfs.write(bytes([b]))
