"""Light-weight job dispatcher (no torch import), for parent processes that
only launch isolated child runs. Same behaviour as common.dispatch: each job
runs in its own subprocess; a failed job is retried in a fresh process, and
the last attempt runs on the CPU (ARC_FORCE_CPU=1). Keeping torch out of the
parent saves about 1.5 GB of committed memory on Windows, which matters when
the machine is shared with other work."""
import os
import subprocess
import sys
import time


def dispatch(script, base_args, jobs, log_prefix="", retries=3, wait_s=60):
    failed = []
    for job in jobs:
        ok = False
        for attempt in range(retries + 1):
            env = dict(os.environ)
            env["ARC_CHILD"] = "1"
            if attempt == retries:
                env["ARC_FORCE_CPU"] = "1"
            cmd = [sys.executable, script] + list(base_args) + list(job) + ["--child"]
            print(f"{log_prefix}job {' '.join(job)} attempt {attempt + 1}"
                  + (" (CPU)" if attempt == retries else ""), flush=True)
            r = subprocess.run(cmd, env=env)
            if r.returncode == 0:
                ok = True
                break
            print(f"{log_prefix}  exit {r.returncode}; retrying in {wait_s}s", flush=True)
            time.sleep(wait_s)
        if not ok:
            failed.append(job)
    if failed:
        print(f"{log_prefix}FAILED jobs: {failed}", flush=True)
    return failed
