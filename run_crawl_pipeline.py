"""
Pipeline script to execute all crawl scripts in sequence.
Streams subprocess output in real-time so progress is visible.
"""
import sys
import subprocess
import time
import os

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def run_script(script_path: str) -> bool:
    """Run a script as a subprocess, streaming its output line-by-line."""
    print("=" * 60, flush=True)
    print(f"▶ RUNNING: {script_path}", flush=True)
    print("=" * 60, flush=True)
    start_time = time.time()

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],   # -u = unbuffered
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        # Stream output line-by-line
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            print(f"  | {line}", flush=True)

        proc.wait()
    except Exception as e:
        print(f"  ✖ Exception while running {script_path}: {e}", flush=True)
        return False

    duration = time.time() - start_time
    if proc.returncode == 0:
        print(f"✔ SUCCESS: {script_path} ({duration:.1f}s)\n", flush=True)
        return True
    else:
        print(f"✖ FAILED: {script_path} exit code {proc.returncode} ({duration:.1f}s)\n", flush=True)
        return False


def main():
    scripts = [
        "crawl-data/crawl_url_and_name.py",
        "crawl-data/crawl_spec_and_variant.py",
        "crawl-data/crawl_description.py",
        "crawl-data/crawl_policy.py",
        "crawl-data/crawl_faq.py",
    ]

    total_start = time.time()
    results = {}

    for script in scripts:
        if not os.path.exists(script):
            print(f"✖ Error: Script {script} does not exist!", flush=True)
            results[script] = False
            continue

        results[script] = run_script(script)

    total_duration = time.time() - total_start

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("PIPELINE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for script, ok in results.items():
        status = "✔ OK" if ok else "✖ FAIL"
        print(f"  {status}  {script}", flush=True)
    print(f"\nTotal time: {total_duration:.1f}s", flush=True)

    if all(results.values()):
        print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    else:
        print("PIPELINE COMPLETED WITH ERRORS", flush=True)


if __name__ == "__main__":
    main()
