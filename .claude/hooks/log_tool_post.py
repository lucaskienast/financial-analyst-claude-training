# Runs after every Claude tool call completes.
# Appends a one-line completion entry to the shared hook log so the full
# lifecycle of each tool invocation (pre + post) is traceable.

import sys
import json
import datetime

hook_payload = json.load(sys.stdin)

tool_name = hook_payload.get('tool_name', 'unknown')
timestamp = datetime.datetime.now().strftime('%H:%M:%S')

with open('/tmp/claude-hooks.log', 'a') as log_file:
    log_file.write(f'[POST] {timestamp} tool={tool_name} done\n')
