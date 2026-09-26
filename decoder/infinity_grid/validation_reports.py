"""Persist actual pytest node reports as phases finish; reduce data deterministically."""
from pathlib import Path
import sys
from .canon import canonical_sha256, write_json_atomic
from . import submission as sub


def selected(node, selectors):
    return any(node==s or node.startswith(s+'::') or node.startswith(s+'[') for s in selectors)


class Recorder:
    def __init__(self, root, binding, selectors):
        self.root=Path(root);self.binding=binding;self.selectors=selectors
        self.observed = {}

    def pytest_collection_finish(self, session):
        row={'binding':self.binding,'selectors':self.selectors,'nodes':[item.nodeid for item in session.items]}
        write_json_atomic(self.root/'collections'/(canonical_sha256(self.selectors)+'.json'),row)

    def pytest_runtest_logstart(self, nodeid, location):
        path=self.root/'nodes'/(canonical_sha256(nodeid)+'.json')
        if path.exists():
            old=sub._read(path)
            write_json_atomic(self.root/'prior_reports'/(canonical_sha256(old)+'.json'),old)
        row={'binding':self.binding,'node':nodeid,'phases':{},'finished':False}
        self.observed[nodeid] = row
        write_json_atomic(path,row)

    def pytest_collectreport(self, report):
        if report.failed:
            write_json_atomic(self.root/'collection_errors'/(canonical_sha256(report.nodeid)+'.json'),
                {'binding':self.binding,'node':report.nodeid,'error':str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        p=self.root/'nodes'/(canonical_sha256(report.nodeid)+'.json')
        row=self.observed.setdefault(report.nodeid,
            {'binding':self.binding,'node':report.nodeid,'phases':{},'finished':False})
        if row['binding']!=self.binding:raise RuntimeError('VALIDATION_REPORT_BINDING')
        row['phases'][report.when]={'outcome':report.outcome,'duration_seconds':report.duration,
            'stdout':report.capstdout,'stderr':report.capstderr,'detail':str(report.longrepr) if report.failed else '',
            'xfail':getattr(report,'wasxfail',None)}
        row['finished']=report.when=='teardown'
        write_json_atomic(p,row)
        sys.stdout.flush();sys.stderr.flush()


    def finalize(self):
        """Flush observed phases and verify exact readback before worker success.

        This retains the ordinary per-phase writes for interruption recovery.
        It never creates outcomes from pytest's aggregate count or exit status.
        A missing observed phase stays missing; a failed phase stays failed.
        """
        for node, row in sorted(self.observed.items()):
            path=self.root/'nodes'/(canonical_sha256(node)+'.json')
            write_json_atomic(path,row)
            if sub._read(path) != row:
                raise RuntimeError('VALIDATION_REPORT_FINAL_READBACK')


def report_rows(root,binding):
    rows=[]
    for p in (Path(root)/'nodes').glob('*.json'):
        row=sub._read(p)
        if row.get('binding')!=binding or canonical_sha256(row['node'])!=p.stem:
            raise RuntimeError('VALIDATION_REPORT_BINDING')
        rows.append(row)
    return rows


def node_status(row):
    if not row.get('finished'):return 'INTERRUPTED'
    phases=row['phases']
    if any(x['outcome']=='failed' for x in phases.values()):return 'FAIL'
    if any(x['outcome']=='skipped' for x in phases.values()):return 'SKIPPED'
    if set(phases)=={'setup','call','teardown'} and all(x['outcome']=='passed' for x in phases.values()):return 'PASS'
    return 'INCOMPLETE'


def pending_selectors(root,binding,selectors):
    reports={r['node']:r for r in report_rows(root,binding)};known=set();covered=[]
    for p in (Path(root)/'collections').glob('*.json'):
        row=sub._read(p)
        if row['binding']!=binding:raise RuntimeError('VALIDATION_COLLECTION_BINDING')
        known.update(row['nodes']);covered.extend(row['selectors'])
    pending=[]
    for sel in selectors:
        nodes=sorted(n for n in known if selected(n,[sel]))
        if sel in covered and nodes:
            pending.extend(n for n in nodes if not reports.get(n,{}).get('finished'))
        else:pending.append(sel)
    return list(dict.fromkeys(pending))


def reduce_reports(root,binding,selectors,worker_results):
    rows=[]
    for row in report_rows(root,binding):
        if not selected(row['node'],selectors):raise RuntimeError('VALIDATION_UNREGISTERED_REPORT')
        status=node_status(row)
        rows.append({'node':row['node'],'status':status,'return_code':0 if status=='PASS' else 1})
    for selector in selectors:
        if not any(selected(r['node'],[selector]) for r in rows):
            rows.append({'node':selector,'status':'NOT_COMPLETED','return_code':2})
    rows.sort(key=lambda r:r['node'])
    errors=[sub._read(p) for p in (Path(root)/'collection_errors').glob('*.json')]
    clean=rows and all(r['status']=='PASS' for r in rows) and not errors and all(r.get('return_code')==0 for r in worker_results)
    return rows, 'PASS' if clean else 'FAIL', errors
