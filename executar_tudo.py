"""Roda as cinco etapas em ordem."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

for script in sorted(SRC.glob("etapa*.py")):
    print(f"\n=== {script.name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)
