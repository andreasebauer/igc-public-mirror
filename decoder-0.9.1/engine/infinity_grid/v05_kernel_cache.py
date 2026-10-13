from __future__ import annotations

"""Bounded execution-local exact-key caches for Decoder kernel services.

Keys are complete immutable structural values. Python's hash table is only an
index; equality is always the key object's exact equality. No digest establishes
scientific equality. Caches are execution-only and can change performance only.
"""

from collections import OrderedDict
from dataclasses import fields, is_dataclass
import sys
from typing import Any


class ExactKernelCacheError(RuntimeError):
    pass


def accounted_bytes(obj: Any) -> int:
    """Conservative retained-object accounting for one cache entry."""
    seen:set[int]=set(); stack=[obj]; total=0
    while stack:
        x=stack.pop(); ident=id(x)
        if ident in seen: continue
        seen.add(ident); total += sys.getsizeof(x)
        if is_dataclass(x):
            stack.extend(getattr(x,f.name) for f in fields(x))
            if hasattr(x,'__dict__'): stack.append(x.__dict__)
        elif isinstance(x,(tuple,list,set,frozenset)):
            stack.extend(x)
        elif isinstance(x,dict):
            stack.extend(x.keys()); stack.extend(x.values())
    return total + 128


class ExactBoundedLRU:
    """Small exact-key LRU with explicit entry/byte bounds and metrics."""
    __slots__=('name','max_entries','max_bytes','_store','_bytes','_stats')

    def __init__(self,name:str,*,max_entries:int,max_bytes:int)->None:
        if not isinstance(name,str) or not name: raise ExactKernelCacheError('CACHE_NAME')
        if type(max_entries) is not int or type(max_bytes) is not int or max_entries<0 or max_bytes<0:
            raise ExactKernelCacheError('CACHE_LIMIT')
        self.name=name; self.max_entries=max_entries; self.max_bytes=max_bytes
        self._store:OrderedDict[Any,tuple[Any,int]]=OrderedDict(); self._bytes=0
        self._stats={'hits':0,'misses':0,'puts':0,'evictions':0,'uncached_oversize':0}

    def lookup(self,key:Any)->tuple[bool,Any]:
        try:
            row=self._store.get(key)
        except TypeError as exc:
            raise ExactKernelCacheError('CACHE_KEY_NOT_HASHABLE') from exc
        if row is None:
            self._stats['misses']+=1; return False,None
        # OrderedDict lookup first uses hash and then exact key equality.
        self._stats['hits']+=1; self._store.move_to_end(key); return True,row[0]

    def put(self,key:Any,value:Any)->bool:
        if not self.max_entries or not self.max_bytes: return False
        try: size=accounted_bytes((key,value))
        except Exception as exc: raise ExactKernelCacheError('CACHE_ACCOUNTING') from exc
        if size>self.max_bytes:
            self._stats['uncached_oversize']+=1; return False
        if key in self._store:
            _old,old_size=self._store.pop(key); self._bytes-=old_size
        while self._store and (len(self._store)>=self.max_entries or self._bytes+size>self.max_bytes):
            _k,(_v,old_size)=self._store.popitem(last=False); self._bytes-=old_size; self._stats['evictions']+=1
        self._store[key]=(value,size); self._bytes+=size; self._stats['puts']+=1; return True

    def clear(self)->None:
        self._store.clear(); self._bytes=0

    def metrics(self,prefix:str|None=None)->dict[str,int]:
        p=(prefix or self.name).rstrip('_')+'_'
        return {**{p+k:int(v) for k,v in self._stats.items()},
                p+'entries':len(self._store),p+'accounted_bytes':self._bytes,
                p+'entry_limit':self.max_entries,p+'byte_limit':self.max_bytes}
