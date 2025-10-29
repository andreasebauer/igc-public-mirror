#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%F_%H%M%S)"
OUTDIR="system_snapshots/snap_${TS}"
mkdir -p "$OUTDIR"

h() { printf "\n## %s\n\n" "$1" >> "$OUTDIR/system_report.md"; }
p() { printf "%s\n" "$*" >> "$OUTDIR/system_report.md"; }

REPOROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPOROOT"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "NA")"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo "NA")"

printf "# System Snapshot (%s)\n\n" "$TS" > "$OUTDIR/system_report.md"
p "**Repo:** $REPOROOT"
p "**Git branch:** $GIT_BRANCH"
p "**Git SHA:** $GIT_SHA"

h "OS / Hardware"
{
  echo '### /etc/os-release'
  test -f /etc/os-release && cat /etc/os-release || echo "no /etc/os-release"
  echo
  echo '### Kernel'
  uname -a || true
  echo
  echo '### CPU'
  (command -v lscpu && lscpu) || echo "lscpu not available"
  echo
  echo '### Memory'
  (command -v free && free -h) || echo "free not available"
  echo
  echo '### Disk (.)'
  df -h . || true
} | sed 's/^/    /' >> "$OUTDIR/system_report.md"

h "Python / venv"
PY_BIN="$(command -v python3 || true)"
PIP_BIN="$(command -v pip || true)"
p "**VIRTUAL_ENV:** ${VIRTUAL_ENV:-<none>}"
p "**python:** ${PY_BIN:-not found}"
p "**pip:** ${PIP_BIN:-not found}"
{
  echo "### Versions"
  (python3 -V 2>&1 || true)
  (pip -V 2>&1 || true)
} | sed 's/^/    /' >> "$OUTDIR/system_report.md"

h "pip packages"
(pip list --format=columns || true) > "$OUTDIR/pip_list.txt"
(pip freeze || true) > "$OUTDIR/requirements-lock.txt"
p "- Saved **pip list** → \`$OUTDIR/pip_list.txt\`"
p "- Saved **pip freeze** (lock) → \`$OUTDIR/requirements-lock.txt\`"
if command -v pipdeptree >/dev/null 2>&1; then
  pipdeptree > "$OUTDIR/pip_deptree.txt" || true
  p "- Saved **pipdeptree** → \`$OUTDIR/pip_deptree.txt\`"
else
  p "- pipdeptree not installed (optional): \`pip install pipdeptree\`"
fi

h "Key Python libraries (NumPy/SciPy/pyFFTW/GUDHI/FastAPI/uvicorn)"
# Write JSON by redirecting stdout of python
python3 - <<'PY' > "$OUTDIR/python_libs.json"
import json, importlib, platform, io, contextlib
def v(name):
    try:
        m=importlib.import_module(name)
        return getattr(m,"__version__","unknown")
    except Exception:
        return None
def numpy_cfg():
    try:
        import numpy
        from numpy.__config__ import show as showcfg
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            showcfg()
        return {"version": numpy.__version__, "show_config": buf.getvalue()}
    except Exception as e:
        return {"error": str(e)}
def ff_tw():
    try:
        import pyfftw
        from pyfftw.interfaces import cache
        return {"version": getattr(pyfftw,"__version__","unknown"),
                "wisdom_cache_enabled": getattr(cache,"enabled",lambda: None)()}
    except Exception:
        return None
data = {
  "python": platform.python_version(),
  "platform": platform.platform(),
  "libraries": {
    "numpy": numpy_cfg(),
    "scipy": v("scipy"),
    "gudhi": v("gudhi"),
    "pyfftw": ff_tw(),
    "fastapi": v("fastapi"),
    "uvicorn": v("uvicorn"),
    "jinja2": v("jinja2"),
    "pydantic": v("pydantic"),
  }
}
print(json.dumps(data, indent=2))
PY

p "- Saved **python libs JSON** → \`$OUTDIR/python_libs.json\`"

h "Python libs summary (excerpt)"
# Use an unquoted heredoc so $OUTDIR expands inside the Python code
python3 - <<PY | sed 's/^/    /' >> "$OUTDIR/system_report.md"
import json, pathlib, sys
path = pathlib.Path("${OUTDIR}/python_libs.json")
try:
    d = json.loads(path.read_text())
except Exception as e:
    print("ERROR reading", path, ":", e)
    sys.exit(0)
print("Python:", d.get("python"))
libs = d.get("libraries", {})
for k,v in libs.items():
    if isinstance(v, dict) and "version" in v:
        print(f"{k}: {v.get('version')}")
    else:
        print(f"{k}: {v}")
PY

h "ASGI Server"
{
  echo "### Uvicorn"
  (uvicorn --version 2>&1 || python3 -c "import uvicorn; print('uvicorn', getattr(uvicorn,'__version__','unknown'))" || true)
} | sed 's/^/    /' >> "$OUTDIR/system_report.md"

h "PostgreSQL"
{
  (psql --version 2>&1 || true)
  (pg_config --version 2>&1 || echo "pg_config not available")
} | sed 's/^/    /' >> "$OUTDIR/system_report.md"

h "Env (safe subset)"
(env | egrep -i '^(VIRTUAL_ENV|PATH|LD_LIBRARY_PATH|PKG_CONFIG_PATH|CFLAGS|LDFLAGS)=' || true) | sed 's/^/    /' >> "$OUTDIR/system_report.md"

h "Git status (porcelain)"
(git status --porcelain=v1 || true) | sed 's/^/    /' >> "$OUTDIR/system_report.md"

echo "✔ Wrote snapshot to: $OUTDIR"
echo "   - $OUTDIR/system_report.md"
echo "   - $OUTDIR/requirements-lock.txt"
echo "   - $OUTDIR/pip_list.txt"
echo "   - $OUTDIR/python_libs.json"
