"""Tidy Kin's memory now (normally a nightly job). No robot needed.

usage: python scripts/consolidate.py [--dry-run]
"""

import json
import sys

from kin.consolidate import consolidate
from kin.memory import Memory

dry_run = "--dry-run" in sys.argv
memory = Memory()
report = consolidate(memory, dry_run=dry_run)

for op in report.operations:
    print(json.dumps(op, ensure_ascii=False))
for note in report.skipped:
    print("skipped:", note)
print(("would have: " if dry_run else "") + report.summary())
