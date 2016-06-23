"""General compression based on probabilistic models.

Usage: z.py compress INFILE OUTFILE [-m MODEL] [-d DEPTH]
       z.py uncompress INFILE OUTFILE [-m MODEL] [-d DEPTH]

Options:
-m MODEL      Prediction model [default: CTW]
-d DEPTH      Depth parameter used for CTW [default: 48]
"""

import model
import ac

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

import fast as modelfast
        
if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)

    depth = int(args['-d'])
    
    if args['-m'] == "CTW":
        probmodel = model.CTW(depth)
    elif args['-m'] == "FastCTW":
        probmodel = modelfast.CTW_KT(depth)
    elif args['-m'] == "PTW:FastCTW":
        probmodel = model.PTW(lambda: modelfast.CTW_KT(depth))
    elif args['-m'] == "FMN:FastCTW":
        probmodel = model.FMN(lambda: modelfast.CTW_KT(depth))
    elif args['-m'] == "CTW_KT":
        probmodel = model.CTW_KT(depth)
    elif args['-m'] == "KT":
        probmodel = model.KT()
    elif args['-m'] == "PTW":
        probmodel = model.PTW()
    elif args['-m'] == "FMN":
        probmodel = model.FMN()
    elif args['-m'] == "PTW:CTW":
        probmodel = model.CommonHistory(lambda history: model.PTW(lambda: model.CTW(depth, history = history)))
    elif args['-m'] == "FMN:CTW":
        probmodel = model.CommonHistory(lambda history: model.FMN(lambda: model.CTW(depth, history = history)))
    elif args['-m'] == "CTW:PTW":
        probmodel = model.CTW(depth, lambda: model.PTW())
    elif args['-m'] == "CTW:FMN":
        probmodel = model.CTW(depth, lambda: model.FMN())
    else:
        raise Error()

    infile = args['INFILE']
    outfile = args['OUTFILE']
        
    if args['compress']:
        msglen = os.path.getsize(infile)
        codelen = 0

        print("Compressing {} ({} bytes)\n".format(infile, msglen))
        start_time = time.time()
        with open(infile, 'rb') as infs, open(outfile, 'wb') as outfs:
            outfs.write(msglen.to_bytes(4, sys.byteorder))
            
            for b in ac.compress_bytes(probmodel, _bytes_with_progress(infs, msglen)):
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
        with open(infile, 'rb') as infs, open(outfile, 'wb') as outfs:
            codelen = os.path.getsize(infile)
            msglen = int.from_bytes(infs.read(4), sys.byteorder)

            for b in ac.decompress_bytes(model, _bytes_with_rprogress(infs), codelen):
                outfs.write(b)
