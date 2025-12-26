import datetime as dt
import json
import os
import logging
import logging.config

from typing_extensions import override
# from typing import override


# Standard LogRecord attributes to exclude when identifying user-provided 'extra' fields
LOG_RECORD_BUILTIN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg",
    "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
}


class JSONFormatter(logging.Formatter):
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
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
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


class ReadableColorFormatter_tester_v2(logging.Formatter):
    """Colorful formatter with optional highlighted 'path' extra field."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    TIMESTAMP = "\x1b[38;2;100;110;130m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m"
    INFO_COLOR = "\x1b[38;2;120;220;180m"
    WARNING_COLOR = "\x1b[38;2;255;200;100m"
    ERROR_COLOR = "\x1b[38;2;255;120;130m"
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"

    FUNC_COLOR = "\x1b[38;2;180;220;255m"
    FILE_COLOR = "\x1b[38;38;2;255;180;120m"
    LINE_COLOR = "\x1b[38;2;140;200;140m"
    AT_COLOR = "\x1b[38;2;160;160;160m"

    PATH_COLOR = "\x1b[38;2;120;200;255m"   
    EXTRA_COLOR = "\x1b[38;2;140;140;140m"
    EXCEPTION = "\x1b[38;2;255;100;100m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: DEBUG_COLOR,
        logging.INFO: INFO_COLOR,
        logging.WARNING: WARNING_COLOR,
        logging.ERROR: ERROR_COLOR,
        logging.CRITICAL: CRITICAL_COLOR,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)
        timestamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        header = f"{self.TIMESTAMP}{timestamp}{self.RESET} | {color}{record.levelname:<8}{self.RESET}"

        location = (
            f"{self.AT_COLOR}[{self.RESET}"
            f"{self.FUNC_COLOR}{record.funcName}(){self.RESET}"
            f"{self.AT_COLOR} @ {self.RESET}"
            f"{self.FILE_COLOR}{record.filename}{self.RESET}"
            f"{self.AT_COLOR}:{self.RESET}"
            f"{self.LINE_COLOR}{record.lineno}{self.RESET}"
            f"{self.AT_COLOR}]{self.RESET}"
        )

        message = record.getMessage()

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._BUILTIN_ATTRS and not k.startswith("_")
        }

        path_str = ""
        if "path" in extras:
            path_val = extras.pop("path")
            path_str = f" {self.PATH_COLOR}{path_val}{self.RESET}"

        extra_str = f" {self.EXTRA_COLOR}>> {extras}{self.RESET}" if extras else ""

        formatted = f"{header} | {location} {message}{path_str}{extra_str}"

        if record.exc_info:
            formatted += f"\n{self.EXCEPTION}{self.formatException(record.exc_info)}{self.RESET}"

        return formatted


class ReadableColorFormatter_tester(logging.Formatter):
    """Colorful formatter with separate colors for funcName, filename, and lineno."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    # Modern color palette
    TIMESTAMP = "\x1b[38;2;100;110;130m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m"
    INFO_COLOR = "\x1b[38;2;120;220;180m"
    WARNING_COLOR = "\x1b[38;2;255;200;100m"
    ERROR_COLOR = "\x1b[38;2;255;120;130m"
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"

    # Separate colors for location parts
    FUNC_COLOR = "\x1b[38;2;180;220;255m"   # Light blue
    FILE_COLOR = "\x1b[38;2;255;180;120m"   # Warm orange
    LINE_COLOR = "\x1b[38;2;140;200;140m"   # Soft green
    AT_COLOR = "\x1b[38;2;160;160;160m"     # Gray for separators

    EXTRA = "\x1b[38;2;140;140;140m"
    EXCEPTION = "\x1b[38;2;255;100;100m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: DEBUG_COLOR,
        logging.INFO: INFO_COLOR,
        logging.WARNING: WARNING_COLOR,
        logging.ERROR: ERROR_COLOR,
        logging.CRITICAL: CRITICAL_COLOR,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)
        timestamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        header = f"{self.TIMESTAMP}{timestamp}{self.RESET} | {color}{record.levelname:<8}{self.RESET}"

        # Separate coloring: [func() @ file.py:lineno]
        location = (
            f"{self.AT_COLOR}[{self.RESET}"
            f"{self.FUNC_COLOR}{record.funcName}(){self.RESET}"
            f"{self.AT_COLOR} @ {self.RESET}"
            f"{self.FILE_COLOR}{record.filename}{self.RESET}"
            f"{self.AT_COLOR}:{self.RESET}"
            f"{self.LINE_COLOR}{record.lineno}{self.RESET}"
            f"{self.AT_COLOR}]{self.RESET}"
        )

        message = record.getMessage()

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._BUILTIN_ATTRS and not k.startswith("_")
        }
        extra_str = f" {self.EXTRA}>> {extras}{self.RESET}" if extras else ""

        formatted = f"{header} | {location} {message}{extra_str}"

        if record.exc_info:
            formatted += f"\n{self.EXCEPTION}{self.formatException(record.exc_info)}{self.RESET}"

        return formatted


class ReadableColorFormatter(logging.Formatter):
    """A colorful, compact, and readable console log formatter with modern colors."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    # Modern color palette
    TIMESTAMP = "\x1b[38;2;200;200;200m" # "\x1b[38;2;200;200;200m" , "\x1b[38;2;150;150;150m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m" # Soft blue
    INFO_COLOR = "\x1b[32;20m"
    WARNING_COLOR = "\x1b[33;20m"  
    ERROR_COLOR = "\x1b[38;2;255;80;80m"     
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"   # Bright red
    LOCATION = "\x1b[38;2;200;200;200m"
    EXTRA = "\x1b[94m" #"\x1b[38;2;140;140;140m"       # Medium gray
    EXCEPTION = "\x1b[38;2;255;100;100m"   # Bright red for exceptions
    MESSAGE = "\x1b[38;2;150;150;150m"   # Medium gray for message
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

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)
        timestamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        header = f"{self.TIMESTAMP}{timestamp}{self.RESET} | {color}{record.levelname:<5}{self.RESET}"

        location = (
            f"{self.LOCATION}[{record.funcName}(){self.RESET}"
            f"{self.AT_COLOR} @ {self.RESET}"
            f"{self.LOCATION}{record.filename}{self.RESET}"
            f"{self.AT_COLOR}:{self.RESET}"
            f"{self.LOCATION}{record.lineno}]{self.RESET}"
        )

        # message = f"{self.MESSAGE}{record.getMessage()}{self.RESET}"
        message = record.getMessage()
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._BUILTIN_ATTRS and not k.startswith("_")
        }

        path_str = ""
        if "path" in extras:
            path_val = extras.pop("path")
            path_str = f" {self.PATH_COLOR}{path_val}{self.RESET}"

        extra_str = f" {self.EXTRA_COLOR}>> {extras}{self.RESET}" if extras else ""

        # formatted = f"{header} | {location} {message}{extra_str}"
        formatted = f"{header} | {location} {message}{path_str}{extra_str}"

        if record.exc_info:
            formatted += f"\n{self.EXCEPTION}{self.formatException(record.exc_info)}{self.RESET}"

        return formatted

    def format_v1(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.DEBUG_COLOR)
        timestamp = dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        header = f"{self.TIMESTAMP}{timestamp}{self.RESET} | {color}{record.levelname:<8}{self.RESET}"
        location = f"{self.LOCATION}{record.name}:{record.lineno}{self.RESET}"
        message = record.getMessage()

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._BUILTIN_ATTRS and not k.startswith("_")
        }
        extra_str = f" {self.EXTRA}>> {extras}{self.RESET}" if extras else ""

        formatted = f"{header} | {location} - {message}{extra_str}"

        if record.exc_info:
            formatted += f"\n{self.EXCEPTION}{self.formatException(record.exc_info)}{self.RESET}"

        return formatted


class ReadableColorFormatter_v1(logging.Formatter):
    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    GREY = "\x1b[38;2;150;150;150m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.GREY)
        timestamp = dt.datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        header = f"{self.GREY}{timestamp}{self.RESET} | {color}{record.levelname:<8}{self.RESET}"
        location = f"{self.BLUE}{record.name}:{record.lineno}{self.RESET}"
        msg = f"{record.getMessage()}"
        
        extras = {k: v for k, v in record.__dict__.items() if k not in self._BUILTIN_ATTRS}
        extra_info = f" {self.GREY}>> {extras}{self.RESET}" if extras else ""

        full_msg = f"{header} | {location} - {msg}{extra_info}"

        if record.exc_info:
            full_msg += f"\n{self.RED}{self.formatException(record.exc_info)}{self.RESET}"
            
        return full_msg


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        # Only allow records with level INFO or lower
        # return record.levelno <= logging.INFO
        # Allow records with level WARNING or lower
        return record.levelno <= logging.WARNING


class ColoredFormatter(logging.Formatter):
    # ANSI Escape Codes
    grey = "\x1b[38;2;150;150;150m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s | %(levelname)-7s | %(message)s"

    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelno) + self.format_str + self.reset
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


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
                    f_out.write(f"{'-'*60}\n\n")
                
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

