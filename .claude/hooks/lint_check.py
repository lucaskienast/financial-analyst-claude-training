# Runs after every Claude tool call.
# When a Write/Edit/MultiEdit produces a .py file, or a NotebookEdit produces an
# .ipynb, this checks that the resulting Python still parses (syntax only, via the
# stdlib compile()). On a SyntaxError it exits with code 2 so the error is fed back
# to Claude to fix. Every other case is a no-op that exits 0. The hook is wrapped so
# that an unexpected failure never blocks the tool pipeline.

import sys
import os
import json
import datetime

LOG_PATH = '/tmp/claude-hooks.log'

WRITE_TOOLS = {'Write', 'Edit', 'MultiEdit'}

hook_payload = json.load(sys.stdin)

tool_name = hook_payload.get('tool_name', '')
tool_input = hook_payload.get('tool_input', {})
file_path = tool_input.get('file_path', '') if isinstance(tool_input, dict) else ''
timestamp = datetime.datetime.now().strftime('%H:%M:%S')


def write_log(message):
    with open(LOG_PATH, 'a') as log_file:
        log_file.write(message + '\n')


def report_invalid(detail):
    # PostToolUse runs after the write, so exiting 2 does not undo the file — it
    # feeds this message back to Claude so it can correct the syntax error.
    write_log(f'[LINT] {timestamp} INVALID {file_path}: {detail}')
    print(f'[LINT] {timestamp} Syntax error in {file_path}: {detail}', file=sys.stderr)
    sys.exit(2)


def check_python_source(source):
    try:
        compile(source, file_path, 'exec')
    except SyntaxError as error:
        offending = (error.text or '').strip()
        report_invalid(f'line {error.lineno}: {error.msg}: {offending}')


def check_notebook(notebook):
    for index, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        raw_source = cell.get('source', '')
        source = ''.join(raw_source) if isinstance(raw_source, list) else raw_source
        lines = source.splitlines()

        # IPython code cells may hold constructs that are not valid Python. Skip
        # whole cells that start with a cell magic (%%...), and drop individual
        # line-magic (%...) and shell-escape (!...) lines, so they do not produce
        # false SyntaxErrors.
        first_line = next((line for line in lines if line.strip()), '')
        if first_line.lstrip().startswith('%%'):
            continue
        kept_lines = [line for line in lines if not line.lstrip().startswith(('%', '!'))]

        cell_name = f'<{os.path.basename(file_path)} cell {index}>'
        try:
            compile('\n'.join(kept_lines), cell_name, 'exec')
        except SyntaxError as error:
            offending = (error.text or '').strip()
            report_invalid(f'cell {index} line {error.lineno}: {error.msg}: {offending}')


if not file_path or not os.path.exists(file_path):
    sys.exit(0)

try:
    if tool_name in WRITE_TOOLS and file_path.endswith('.py'):
        with open(file_path, encoding='utf-8') as source_file:
            check_python_source(source_file.read())
    elif file_path.endswith('.ipynb'):
        with open(file_path, encoding='utf-8') as notebook_file:
            check_notebook(json.load(notebook_file))
    else:
        sys.exit(0)
except SystemExit:
    raise
except Exception as error:
    # Never crash the tool pipeline over an unexpected error (unreadable file,
    # malformed notebook JSON, etc.) — log it and treat as a no-op.
    write_log(f'[LINT] {timestamp} skipped {file_path}: {error!r}')
    sys.exit(0)

write_log(f'[LINT] {timestamp} ok {file_path}')
sys.exit(0)
