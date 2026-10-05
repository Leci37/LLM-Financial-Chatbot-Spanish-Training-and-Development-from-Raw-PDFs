"""Static check of the pipeline's imports. Needs no third-party package installed.

For every .py file in finchat/ and pipeline/:
  - each `finchat...` import must resolve to a file in the repo;
  - nothing may import the pre-2026 module names (_utils, _validation, _1_utils, ...).

finchat.google_generate_ask is reported but does not fail the check: step 2 needs it and it
was never committed (see the README, "Known issues"). Run from anywhere:
    python tools/check_imports.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKED = ["finchat", "pipeline"]
OLD_NAMES = {"_utils", "_validation", "_1_utils", "_google_generate_ask"}
KNOWN_MISSING = {"finchat.google_generate_ask": "step 2's question model; never committed to the repository"}


def resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        base = ROOT.joinpath(*parts[:i])
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


problems, notes = [], set()
files = [p for d in CHECKED for p in (ROOT / d).rglob("*.py")]
for path in files:
    rel = path.relative_to(ROOT)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets = [node.module]
        else:
            continue
        for t in targets:
            if t.split(".")[0] in OLD_NAMES:
                problems.append(f"{rel}:{node.lineno}: imports old module '{t}'")
            elif t in KNOWN_MISSING:
                notes.add(f"note: {t} is missing ({KNOWN_MISSING[t]}); imported by {rel}")
            elif t.split(".")[0] == "finchat" and not resolves(t):
                problems.append(f"{rel}:{node.lineno}: '{t}' does not resolve")

print(f"checked {len(files)} files in {', '.join(CHECKED)}")
print("\n".join(sorted(notes)))
if problems:
    print("\n".join(problems))
    sys.exit(1)
print("all finchat imports resolve (apart from the known gap above); no old module names")
