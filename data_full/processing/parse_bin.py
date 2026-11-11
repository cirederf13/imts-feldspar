"""Parse Risoe BIN v8 files (the *.binx uploads from Aditi) -> list of records."""
import struct, numpy as np

def read_bin(path):
    b = open(path, 'rb').read(); off = 0; out = []
    while off < len(b):
        ver, length, prev, npts = struct.unpack_from('<HIII', b, off)
        assert ver == 8, ver
        rec = b[off:off+length]
        run, set_, pos = struct.unpack_from('<hhh', rec, 15)
        data = np.frombuffer(rec[length-4*npts:], dtype='<i4').astype(float)
        out.append(dict(idx=len(out), run=run, set=set_, pos=pos, npts=npts,
                        tmax=struct.unpack_from('<f', rec, 334)[0],
                        rate=struct.unpack_from('<f', rec, 338)[0],
                        irr=struct.unpack_from('<f', rec, 360)[0],
                        ltype=rec[324], data=data,
                        date=rec[length-4*npts-0:][:0]))
        off += length
    assert off == len(b)
    return out
