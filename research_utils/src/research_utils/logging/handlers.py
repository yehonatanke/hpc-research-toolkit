import datetime as dt
import json
import os
import logging
import logging.config

from typing_extensions import override
from research_utils.core.path import path_with_parents

# from typing import override


# Standard LogRecord attributes to exclude when identifying user-provided 'extra' fields
LOG_RECORD_BUILTIN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg",
    "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
}


class JSONFormatter(logging.Formatter):
    """JSON formatter."""

    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        # Mapping of output JSON keys to LogRecord attribute names
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        # Build the dictionary and serialize to a single-line JSON string
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        # Mandatory fields for every log entry
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }

        # Format exception tracebacks if present
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        # Format stack info if present
        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        # Map LogRecord attributes to user-defined keys from fmt_keys
        message = {}
        for key, val in self.fmt_keys.items():
            msg_val = always_fields.pop(val, None)
            message[key] = msg_val if msg_val is not None else getattr(record, val)

        # Append any remaining mandatory fields (like timestamp/exc_info)
        message.update(always_fields)

        # Automatically include all user-provided 'extra' fields
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class FileFormatter(logging.Formatter):
    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def __init__(self, *, fmt_keys: dict[str, str] | None = None, use_color: bool = False):
        """log formatter for .log files.
        :param use_color: Set to False for .log files to avoid ANSI 'gibberish'.
                          Set to True for terminal output.
        """
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}
        self.use_color = use_color

        codes = {
            "TIMESTAMP": "\x1b[90m",
            "DEBUG_COLOR": "\x1b[34m",
            "INFO_COLOR": "\x1b[32m",
            "WARNING_COLOR": "\x1b[33m",
            "ERROR_COLOR": "\x1b[31m",
            "CRITICAL_COLOR": "\x1b[35m",
            "LOCATION": "\x1b[90m",
            "EXCEPTION": "\x1b[31m",
            "PATH_COLOR": "\x1b[36m",
            "AT_COLOR": "\x1b[37m",
            "EXTRA_COLOR": "\x1b[90m",
            "RESET": "\x1b[0m",
        }

        # Toggle colors based on use_color flag
        for key, code in codes.items():
            setattr(self, key, code if use_color else "")

        self.COLORS = {
            logging.DEBUG: self.DEBUG_COLOR,
            logging.INFO: self.INFO_COLOR,
            logging.WARNING: self.WARNING_COLOR,
            logging.ERROR: self.ERROR_COLOR,
            logging.CRITICAL: self.CRITICAL_COLOR,
        }

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
        }
        if record.exc_info:
            always_fields["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        log_data = {}
        for key, val in self.fmt_keys.items():
            msg_val = always_fields.pop(val, None)
            log_data[key] = msg_val if msg_val is not None else getattr(record, val)

        log_data.update(always_fields)
        for key, val in record.__dict__.items():
            if key not in self._BUILTIN_ATTRS and not key.startswith("_"):
                log_data[key] = val
        return log_data

    def format(self, record: logging.LogRecord) -> str:
        data = self._prepare_log_dict(record)
        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)

        header = f"{self.TIMESTAMP}{data.get('timestamp')}{self.RESET} | {color}{record.levelname:<7}{self.RESET}"

        rel_path = path_with_parents(getattr(record, "pathname", record.filename), 2)

        location = (
            f"{self.LOCATION}[{record.funcName}(){self.RESET}"
            f"{self.AT_COLOR} @ {self.RESET}"
            # f"{self.LOCATION}{record.filename}{self.RESET}"
            f"{self.LOCATION}{rel_path}{self.RESET}"
            f"{self.AT_COLOR}:{self.RESET}"
            f"{self.LOCATION}{record.lineno}]{self.RESET}"
        )

        message = data.get("message")
        path_str = f" {self.PATH_COLOR}{data.pop('path')}{self.RESET}" if "path" in data else ""

        used_keys = {"message", "timestamp", "exc_info", "stack_info"}
        extras = {k: v for k, v in data.items() if k not in used_keys}
        extra_str = f" {self.EXTRA_COLOR}>> {extras}{self.RESET}" if extras else ""

        formatted = f"{header} | {location} {message}{path_str}{extra_str}"

        if "exc_info" in data:
            formatted += f"\n{self.EXCEPTION}{data['exc_info']}{self.RESET}"
        if "stack_info" in data:
            formatted += f"\n{data['stack_info']}"

        return formatted


class ConsoleFormatter(logging.Formatter):
    """fast formatter for console output."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    TIMESTAMP = "\x1b[38;2;200;200;200m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m"
    INFO_COLOR = "\x1b[32;20m"
    WARNING_COLOR = "\x1b[33;20m"
    ERROR_COLOR = "\x1b[38;2;255;80;80m"
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"
    LOCATION = "\x1b[38;2;200;200;200m"
    EXCEPTION = "\x1b[38;2;255;100;100m"
    PATH_COLOR = "\x1b[38;2;120;200;255m"
    AT_COLOR = "\x1b[38;2;160;160;160m"
    EXTRA_COLOR = "\x1b[38;2;140;140;140m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: DEBUG_COLOR,
        logging.INFO: INFO_COLOR,
        logging.WARNING: WARNING_COLOR,
        logging.ERROR: ERROR_COLOR,
        logging.CRITICAL: CRITICAL_COLOR,
    }

    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
        }

        if record.exc_info:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        log_data = {}
        for key, val in self.fmt_keys.items():
            msg_val = always_fields.pop(val, None)
            log_data[key] = msg_val if msg_val is not None else getattr(record, val)

        log_data.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in self._BUILTIN_ATTRS and not key.startswith("_"):
                log_data[key] = val

        return log_data

    def format(self, record: logging.LogRecord) -> str:
        data = self._prepare_log_dict(record)

        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)

        header = f"{self.TIMESTAMP}{data.get('timestamp')}{self.RESET} | {color}{record.levelname}{self.RESET}"

        rel_path = path_with_parents(getattr(record, "pathname", record.filename), 2)

        location = (
            f"{self.LOCATION}[{record.funcName}(){self.RESET}"
            f"{self.AT_COLOR} @ {self.RESET}"
            # f"{self.LOCATION}{record.filename}{self.RESET}"
            f"{self.LOCATION}{rel_path}{self.RESET}"
            f"{self.AT_COLOR}:{self.RESET}"
            f"{self.LOCATION}{record.lineno}]{self.RESET}"
        )

        message = data.get("message")

        # Handle path specifically if it exists in the prepared data
        path_str = ""
        if "path" in data:
            path_val = data.pop("path")
            path_str = f" {self.PATH_COLOR}{path_val}{self.RESET}"

        # Extract remaining extras (exclude already used fields)
        used_keys = {"message", "timestamp", "exc_info", "stack_info"}
        extras = {k: v for k, v in data.items() if k not in used_keys}

        extra_str = f" {self.EXTRA_COLOR}>> {extras}{self.RESET}" if extras else ""

        formatted = f"{header} | {location} {message}{path_str}{extra_str}"

        if "exc_info" in data:
            formatted += f"\n{self.EXCEPTION}{data['exc_info']}{self.RESET}"

        if "stack_info" in data:
            formatted += f"\n{data['stack_info']}"

        return formatted


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        # Only allow records with level INFO or lower
        # return record.levelno <= logging.INFO
        # Allow records with level WARNING or lower
        return record.levelno <= logging.WARNING


def setup_logging(logging_config_path):
    """Initializes logging from a JSON file."""
    if os.path.exists(logging_config_path):
        with open(logging_config_path, "rt") as f:
            config = json.load(f)

        # Ensure log directory exists if a file handler is used
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                log_file = handler["filename"]
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                with open(log_file, "a", encoding="utf-8") as f_out:
                    f_out.write(f"\n{'-'*60}\n")
                    f_out.write(f"START NEW RUN: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f_out.write(f"{'-'*60}\n")

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
