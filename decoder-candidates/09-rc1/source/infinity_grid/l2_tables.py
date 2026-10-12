"""Frozen L2 table I/O, independent of adapter registration.

The loader implementation is preserved from dev88; input semantics are unchanged.
"""
from __future__ import annotations
import array, ast, struct, sys
from pathlib import Path

def _load_npy(path: Path):
    b=path.read_bytes()
    if b[:6] != b'\x93NUMPY': raise ValueError(f'not NPY: {path}')
    major=b[6]
    if major==1:
        hlen=struct.unpack('<H',b[8:10])[0]; off=10
    elif major in (2,3):
        hlen=struct.unpack('<I',b[8:12])[0]; off=12
    else: raise ValueError(f'unsupported NPY version {major}')
    enc='utf-8' if major==3 else 'latin1'
    hdr=ast.literal_eval(b[off:off+hlen].decode(enc).strip())
    if hdr.get('fortran_order'): raise ValueError('Fortran-order NPY unsupported by clean-room L2 replay')
    descr=hdr['descr']; code={'<i2':'h','<i4':'i','|i1':'b','<i1':'b'}.get(descr)
    if code is None: raise ValueError(f'unsupported dtype {descr}')
    a=array.array(code); a.frombytes(b[off+hlen:])
    if sys.byteorder!='little' and descr.startswith('<'): a.byteswap()
    return list(a), tuple(hdr['shape'])

class Tables:
    def __init__(self,root:Path):
        d=root/'data'
        self.tri,self.trish=_load_npy(d/'ig_rm1_L1L1_triples.npy')
        self.cid,_=_load_npy(d/'ig_rm1_L1L1_candidate_id.npy')
        self.oc,_=_load_npy(d/'ig_rm1_opt_count.npy')
        self.ot,_=_load_npy(d/'ig_rm1_opt_target.npy')
        self.os,_=_load_npy(d/'ig_rm1_opt_supply.npy')
        self.om,_=_load_npy(d/'ig_rm1_opt_missing.npy')
        self.on,_=_load_npy(d/'ig_rm1_opt_need.npy')
        self.rk,_=_load_npy(d/'ig_rm1_opt_rank.npy')
        self.lookup={self.triple(i):i for i in range(self.trish[0])}
        self.rep={}
        for i,c in enumerate(self.cid): self.rep.setdefault(c,i)
    def triple(self,tid): return tuple(self.tri[tid*3:tid*3+3])
    @staticmethod
    def at(a,s,i): return a[s*10+i]

