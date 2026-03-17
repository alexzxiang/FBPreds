import sys
print("Starting...", file=sys.stderr, flush=True)
print("Test output")
sys.stderr.flush()
sys.stdout.flush()
print("Done", file=sys.stderr, flush=True)
