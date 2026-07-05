# Runs before every Claude tool call.
# Appends a one-line entry to the shared hook log so every tool invocation
# is visible for debugging and auditing purposes.

import sys
import json
import datetime

hook_payload = json.load(sys.stdin)

tool_name = hook_payload.get('tool_name', 'unknown')
tool_input_preview = str(hook_payload.get('tool_input', ''))[:80]
timestamp = datetime.datetime.now().strftime('%H:%M:%S')

with open('/tmp/claude-hooks.log', 'a') as log_file:
    log_file.write(f'[PRE]  {timestamp} tool={tool_name} input={tool_input_preview}\n')
