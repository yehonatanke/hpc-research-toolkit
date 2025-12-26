import sys
import os
import inspect
from modules.util import print_args_to_log

_LOG_FILE_HANDLER = None
_DEBUG_LEVEL = 0

def setup_logging(log_filepath="debug.log", args=None):
    global _LOG_FILE_HANDLER
    if not log_filepath:
        print(f"\n[INFO] No log file path. Printing to console (sys.stderr).\n")
        return

    try:
        _LOG_FILE_HANDLER = open(log_filepath, 'w')
        print(f"\n[INFO] Logging to: {os.path.abspath(log_filepath)}\n")
        # print_args_to_log(args, _LOG_FILE_HANDLER)
        if args:
            print_args_to_log(args, _LOG_FILE_HANDLER)
    except IOError as e:
        print(f"[Error] Could not open log file {log_filepath}: {e}", file=sys.stderr)
        _LOG_FILE_HANDLER = sys.stderr

def close_logging():
    global _LOG_FILE_HANDLER
    if _LOG_FILE_HANDLER and _LOG_FILE_HANDLER != sys.stdout:
        _LOG_FILE_HANDLER.close()
        _LOG_FILE_HANDLER = None

def set_debug_level(level: int):
    global _DEBUG_LEVEL
    _DEBUG_LEVEL = level

def debug_print(level: int, *args, **kwargs):
    output_stream = _LOG_FILE_HANDLER if _LOG_FILE_HANDLER else sys.stderr
    if _DEBUG_LEVEL >= level:
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        prefix = f"[DEBUG] [{func_name}() @ {filename}:{lineno}]"
        print(prefix, *args, **kwargs, file=output_stream)
        if output_stream == sys.stderr:
            output_stream.flush()