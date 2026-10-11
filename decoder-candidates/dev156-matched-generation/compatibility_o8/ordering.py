"""Historical execution ordering metadata; native V3 object identities stay intact."""
import hashlib,json
def historical_order_key(state):
    if state.level==7:
        payload={'level':7,'parents':list(state.ctx.source_order),
                 'edges':[list(e) for e in state.edges],
                 'reserve_counts':list(state.reserve_counts),'source':state.source_id}
    else:
        payload={'level':state.level,'children':[historical_order_key(c) for c in state.children],
                 'edges':[[u,v,a,b,path_json(wu),path_json(wv)] for u,v,a,b,wu,wv in state.top_edges_full],
                 'lane':state.lane,'motif':state.motif_id}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def path_json(x):
    return [path_json(v) for v in x] if isinstance(x,(tuple,list)) else int(x)
class _OrderingView:
    def __init__(self,state):self.state=state;self.construction_digest=historical_order_key(state)
    def __getattr__(self,name):return getattr(self.state,name)
def select_with_historical_order(scanner,states,k,must_include=None):
    views=[_OrderingView(s) for s in states]
    required=[_OrderingView(s) for s in must_include or []]
    return [v.state for v in scanner._farthest_select(views,k,required)]
