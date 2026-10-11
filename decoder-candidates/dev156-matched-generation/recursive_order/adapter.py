"""Explicit historical execution routing with native V3 state construction."""
from ordering import historical_order_key
def reserve_with_historical_order(state,t,scanner):
    if state.level==7:return state.reserve_external(t)
    eligible=[(historical_order_key(c),i) for i,c in enumerate(state.children) if c.total_caps[t]>0]
    if not eligible:raise RuntimeError(f'O{state.level} no endpoint type {t}')
    _,i=min(eligible)
    child,witness=reserve_with_historical_order(state.children[i],t,scanner)
    children=list(state.children);children[i]=child
    return scanner.LiftState(state.engine,state.level,tuple(children),state.top_edges_full,state.lane,state.motif_id),(i,witness,t)
