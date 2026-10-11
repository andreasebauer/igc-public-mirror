"""Process-local historical execution policy; preserve native V3 identity."""
from contextlib import contextmanager
from adapter import reserve_with_historical_order
@contextmanager
def historical_reservation_policy(scanner):
    original=scanner.LiftState.reserve_external
    def reserve(state,t):
        cache=state.__dict__.setdefault('_historical_policy_reserve_cache',{})
        if t not in cache:cache[t]=reserve_with_historical_order(state,t,scanner)
        return cache[t]
    scanner.LiftState.reserve_external=reserve
    try:yield
    finally:scanner.LiftState.reserve_external=original
