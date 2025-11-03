#!/usr/bin/env bash
set -euo pipefail
STAMP="$(date +%F_%H%M%S)"
OUTDIR="${HOME}/igc/system_snapshots"
OUT="${OUTDIR}/system_report_${STAMP}.md"
mkdir -p "${OUTDIR}"
REPO="${HOME}/igc"
if git -C "${REPO}" rev-parse --git-dir >/dev/null 2>&1; then
  GIT_BRANCH="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD || true)"
  GIT_SHA="$(git -C "${REPO}" rev-parse HEAD || true)"
  GIT_STATUS="$(git -C "${REPO}" status --porcelain=v1 || true)"
  GIT_LAST_COMMIT="$(git -C "${REPO}" log -1 --pretty=fuller || true)"
else
  GIT_BRANCH="(no git repo found)"; GIT_SHA="(n/a)"; GIT_STATUS=""; GIT_LAST_COMMIT=""
fi
if [[ -d "${REPO}/.venv" ]]; then
  PY="${REPO}/.venv/bin/python3"; PIP="${REPO}/.venv/bin/pip"
else
  PY="$(command -v python3 || true)"; PIP="$(command -v pip3 || command -v pip || true)"
fi
OS_RELEASE="$(cat /etc/os-release 2>/dev/null || true)"
KERNEL="$(uname -a || true)"
LSCPU="$(/usr/bin/lscpu 2>/dev/null || true)"
MEMORY="$(/usr/bin/free -h 2>/dev/null || true)"
DISK_ROOT="$(df -h / 2>/dev/null || true)"
DISK_HOME="$(df -h ~ 2>/dev/null || true)"
if [[ -n "${PIP}" ]]; then
  PIP_LIST="$("${PIP}" list --format=columns 2>&1 || true)"
  PIP_FREEZE="$("${PIP}" freeze 2>&1 || true)"
else
  PIP_LIST="pip not found"; PIP_FREEZE="pip not found"
fi
if [[ -n "${PY}" ]]; then
  PY_LIBS_JSON="$("${PY}" - <<'PY' 2>&1 || true
import json, platform, sys, subprocess
def safe_run(cmd):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as e: return f"ERROR: {e}"
def pkg_ver(n):
    try: import importlib.metadata as m; return m.version(n)
    except Exception: return None
libs={"numpy":{"version":pkg_ver("numpy"),
"show_config":safe_run([sys.executable,"-c","import numpy as np; np.show_config()"])},
"scipy":pkg_ver("scipy"),"gudhi":pkg_ver("gudhi"),
"pyfftw":{"version":pkg_ver("pyFFTW"),"wisdom_cache":None},
"fastapi":pkg_ver("fastapi"),"uvicorn":pkg_ver("uvicorn"),
"jinja2":pkg_ver("Jinja2"),"pydantic":pkg_ver("pydantic")}
out={"python":platform.python_version(),"platform":platform.platform(),
"executable":sys.executable,"libraries":libs}
print(json.dumps(out,indent=2))
PY
)"
else
  PY_LIBS_JSON="python3 not found"
fi
cat > "${OUT}" <<EOT
# System Snapshot — ${STAMP}

**Repo:** ${REPO}  
**Git branch:** ${GIT_BRANCH}  
**Git SHA:** ${GIT_SHA}  

---

## OS / Hardware
### /etc/os-release
\`\`\`text
${OS_RELEASE}
\`\`\`
### Kernel
\`\`\`text
${KERNEL}
\`\`\`
### CPU
\`\`\`text
${LSCPU}
\`\`\`
### Memory
\`\`\`text
${MEMORY}
\`\`\`
### Disk (root /)
\`\`\`text
${DISK_ROOT}
\`\`\`
### Disk (home ~)
\`\`\`text
${DISK_HOME}
\`\`\`

---

## Git
### Last commit
\`\`\`text
${GIT_LAST_COMMIT}
\`\`\`
### Status
\`\`\`text
${GIT_STATUS}
\`\`\`

---

## Python / pip
**VIRTUAL_ENV:** ${VIRTUAL_ENV:-"(not active or using system python)"}  
**python:** ${PY:-"(not found)"}  
**pip:** ${PIP:-"(not found)"}  

### pip list
\`\`\`text
${PIP_LIST}
\`\`\`
### pip freeze
\`\`\`text
${PIP_FREEZE}
\`\`\`
### Key libraries (JSON summary + numpy.show_config)
\`\`\`json
${PY_LIBS_JSON}
\`\`\`
EOT
echo "✅ Snapshot written: ${OUT}"
