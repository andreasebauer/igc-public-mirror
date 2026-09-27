"""Generate a reviewable, non-activating host install packet for an exact commit."""
import argparse
import json
from pathlib import Path
import re


def render(commit,output):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact 40-character commit required')
    output=Path(output)
    if output.exists():raise ValueError('Output must be a new directory')
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output prohibited')
    release='/opt/infinity-grid/releases/'+commit;repo=release+'/repo';state='/var/lib/infinity-grid';config='/etc/infinity-grid/config.json'
    settings={'engine_repository':repo,'engine_python':release+'/engine-venv/bin/python',
              'workspace_root':state+'/workspaces','capture_store':state+'/workspaces/store',
              'specification_root':state+'/specifications','catalog':state+'/catalog.json',
              'worker_state':state+'/web-state','allowed_hosts':['127.0.0.1','localhost'],
              'drive_token_file':None,'drive_folder_id':None}
    files={'config.json':json.dumps(settings,indent=2)+'\n','catalog.json':'{"jobs":[],"tasks":[],"inputs":[]}\n'}
    for role in ('api','execution','control','save'):
        credential='LoadCredential=web-token:/etc/infinity-grid/web-token\n' if role=='api' else ''
        files['ig-decoder-'+role+'.service']=f'''[Unit]
Description=Infinity Grid Decoder {role}
After=network-online.target
Wants=network-online.target
RequiresMountsFor={state} {release}

[Service]
Type=simple
User=igdecoder
Group=igdecoder
WorkingDirectory={repo}/decoder-web
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONNOUSERSITE=1
UMask=0077
{credential}ExecStart={release}/web-venv/bin/python -B -m ig_web.service --config {config} --role {role}
Restart=on-failure
RestartSec=10
KillMode=mixed
SendSIGKILL=no
TimeoutStopSec=infinity
RuntimeMaxSec=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
'''
    files['install.sh']=f'''#!/bin/sh
set -eu
# Fresh installation only. Does not enable/start services or activate science.
test "$(id -u)" = 0 || {{ echo 'Run installation as root'; exit 1; }}
test "$(uname -s)" = Linux
python3.12 -c 'import sys,platform; assert sys.version_info[:2]==(3,12) and platform.python_implementation()=="CPython"'
command -v git >/dev/null
command -v systemctl >/dev/null
test ! -e {release}
test ! -e /etc/infinity-grid/config.json
test ! -e /etc/systemd/system/ig-decoder-api.service
python3.12 - <<'PY'
from pathlib import Path
for name in ('{release}','{state}','/etc/infinity-grid','/etc/systemd/system'):
    p=Path(name)
    assert not any(x.is_symlink() for x in (p,*p.parents)), 'Symlink installation path refused'
PY
packet=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
getent passwd igdecoder >/dev/null || useradd --system --user-group --home-dir {state} --shell /usr/sbin/nologin igdecoder
install -d -m 0755 {release}
git clone --no-checkout https://github.com/andreasebauer/igc-public-mirror.git {repo}
git -C {repo} fetch origin {commit}
git -C {repo} checkout --detach {commit}
test "$(git -C {repo} rev-parse HEAD)" = {commit}
git -C {repo} diff --exit-code
python3.12 {repo}/decoder-import/verify_source.py
python3.12 -m venv {release}/engine-venv
{release}/engine-venv/bin/python -m pip install --require-hashes -r {repo}/decoder/qualification/requirements-linux-py312.txt
python3.12 -m venv {release}/web-venv
{release}/web-venv/bin/python -m pip install -r {repo}/decoder-web/deploy/bootstrap.lock.txt
{release}/web-venv/bin/python -m pip install -r {repo}/decoder-web/requirements.lock.txt
{release}/web-venv/bin/python -m pip install --no-deps --no-build-isolation {repo}/decoder-web
install -d -m 0700 -o igdecoder -g igdecoder {state} {state}/workspaces {state}/workspaces/store {state}/workspaces/projects {state}/specifications {state}/web-state
install -d -m 0750 -o root -g igdecoder /etc/infinity-grid
test -e {state}/catalog.json || install -m 0600 -o igdecoder -g igdecoder "$packet/catalog.json" {state}/catalog.json
install -m 0640 -o root -g igdecoder "$packet/config.json" /etc/infinity-grid/config.json
for role in api execution control save; do
    test ! -e "/etc/systemd/system/ig-decoder-$role.service"
    install -m 0644 "$packet/ig-decoder-$role.service" "/etc/systemd/system/ig-decoder-$role.service"
done
systemctl daemon-reload
runuser -u igdecoder -- {release}/web-venv/bin/python -B -m ig_web.service --config {config} --role execution --check
echo 'Installation staged. Services are NOT enabled or started. Continue with host verification and Block 5 Step 2.'
'''
    files['RELEASE.json']=json.dumps({'commit':commit,'release':release,'state':state,
        'status':'INSTALL_PACKET_ONLY_NOT_DEPLOYED','services_started':False},indent=2)+'\n'
    output.mkdir(parents=True,mode=0o700)
    for name,value in files.items():(output/name).write_text(value)
    return files


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();render(args.commit,args.output)
