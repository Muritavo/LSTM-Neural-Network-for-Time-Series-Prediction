import time
import sys, json

while True:
    if not sys.stdin.isatty():
        break
    time.sleep(1)

sys.stdout.write(json.load(sys.stdin)["CONFIG"])