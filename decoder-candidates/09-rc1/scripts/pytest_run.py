from pathlib import Path
import sys
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root/'source'))
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
