# Runs before every Claude tool call.
# Exits with code 2 (blocking the tool) if a Bash command matches known
# destructive patterns such as recursive deletes, disk writes, or world-writable
# chmod — things that could cause irreversible damage to the local filesystem.

import sys
import json
import datetime
import re

DANGEROUS_PATTERN = re.compile(
    r'rm -rf|rimraf /|dd if=|mkfs\.|chmod -R 777|truncate -s 0'
)

hook_payload = json.load(sys.stdin)

tool_name = hook_payload.get('tool_name', '')
tool_input = hook_payload.get('tool_input', {})
bash_command = tool_input.get('command', '') if isinstance(tool_input, dict) else ''
timestamp = datetime.datetime.now().strftime('%H:%M:%S')

is_bash_tool = tool_name == 'Bash'
is_dangerous = bool(DANGEROUS_PATTERN.search(bash_command))

if is_bash_tool and is_dangerous:
    blocked_message = f'[BLOCKED] {timestamp} Dangerous command rejected: {bash_command}'
    with open('/tmp/claude-hooks.log', 'a') as log_file:
        log_file.write(blocked_message + '\n')
    print(blocked_message, file=sys.stderr)
    sys.exit(2)
