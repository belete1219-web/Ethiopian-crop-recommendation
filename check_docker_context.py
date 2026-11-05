"""Static check of the Docker build context (works without Docker installed).

It does NOT build the image. It checks that every file the Dockerfile COPYs exists, is not excluded by
.dockerignore, that the Dockerfile pins the same versions as requirements.txt, and that the API's imports are
covered by requirements-deploy.txt. Run: python check_docker_context.py
"""
import fnmatch
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
problems = []

ignore = [l.strip() for l in (root / ".dockerignore").read_text().splitlines() if l.strip() and not l.startswith("#")]


def ignored(rel: str) -> bool:
    parts = rel.split("/")
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[0], pat) or fnmatch.fnmatch(parts[-1], pat)
               for pat in ignore)


copied = []
for line in (root / "Dockerfile").read_text().splitlines():
    m = re.match(r"\s*COPY\s+(.+)", line)
    if m:
        *srcs, _dest = m.group(1).split()
        copied += srcs
for src in copied:
    ok = (root / src).exists()
    ign = ignored(src)
    print(f"[{'PASS' if ok and not ign else 'FAIL'}] COPY {src}: exists={ok}, excluded_by_dockerignore={ign}")
    if not ok or ign:
        problems.append(src)


def pins(path):
    return {m.group(1).lower(): m.group(2) for m in
            re.finditer(r"^([A-Za-z0-9_.-]+)==([0-9][^\s#]*)", (root / path).read_text(), re.M)}


full, deploy = pins("requirements.txt"), pins("requirements-deploy.txt")
for pkg, ver in deploy.items():
    same = full.get(pkg) == ver
    print(f"[{'PASS' if same else 'FAIL'}] {pkg}=={ver} matches requirements.txt ({full.get(pkg)})")
    if not same:
        problems.append(pkg)

deploy_text = (root / "requirements-deploy.txt").read_text().lower()
needed = {"fastapi": "fastapi", "uvicorn": "uvicorn", "joblib": "joblib", "numpy": "numpy", "pandas": "pandas",
          "sklearn": "scikit-learn"}
for mod, pkg in needed.items():
    ok = pkg in deploy_text
    print(f"[{'PASS' if ok else 'FAIL'}] runtime package for import '{mod}' listed in requirements-deploy.txt")
    if not ok:
        problems.append(pkg)

print("\nSTATIC CHECK PASSED (image NOT built)" if not problems else f"\nPROBLEMS: {problems}")
sys.exit(1 if problems else 0)
