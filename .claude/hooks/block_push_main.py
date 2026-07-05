# Runs before every Claude tool call.
# Exits with code 2 (blocking the tool) if a Bash command would push to the
# main or master branch, or uses --force on any push.
# This prevents accidental direct pushes to protected branches.

import sys
import json
import datetime
import re

PROTECTED_PUSH_PATTERN = re.compile(
    r'git push.*(main|master)|git push.*--force'
)

hook_payload = json.load(sys.stdin)

tool_name = hook_payload.get('tool_name', '')
tool_input = hook_payload.get('tool_input', {})
bash_command = tool_input.get('command', '') if isinstance(tool_input, dict) else ''
timestamp = datetime.datetime.now().strftime('%H:%M:%S')

is_bash_tool = tool_name == 'Bash'
is_protected_push = bool(PROTECTED_PUSH_PATTERN.search(bash_command))

if is_bash_tool and is_protected_push:
    blocked_message = f'[BLOCKED] {timestamp} Push to main/master blocked: {bash_command}'
    with open('/tmp/claude-hooks.log', 'a') as log_file:
        log_file.write(blocked_message + '\n')
    print(blocked_message, file=sys.stderr)
    sys.exit(2)
