#!/usr/bin/env python3
"""Require the next fixed step; allow an honest infrastructure-error report."""
import json
from _common import active, event, status

if __name__ == "__main__":
    ev = event()
    run, st = active()
    if run and st.get("phase") not in ("done", "done_with_limitations", "aborted", "error", "interrupted") and not ev.get("stop_hook_active"):
        print(json.dumps({"decision": "block", "reason": "The fixed pipeline is unfinished. Follow its next action; do not invent or skip stages.\n" + status()}))
