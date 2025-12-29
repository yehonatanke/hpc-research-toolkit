

class ReadableColorFormatter_tester_v2(logging.Formatter):
    """log formatter."""

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


class ReadableColorFormatter_v3(logging.Formatter):
    """slow log formatter."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    TIMESTAMP = "\x1b[38;2;200;200;200m" # "\x1b[38;2;200;200;200m" , "\x1b[38;2;150;150;150m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m" 
    INFO_COLOR = "\x1b[32;20m"
    WARNING_COLOR = "\x1b[33;20m"  
    ERROR_COLOR = "\x1b[38;2;255;80;80m"     
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"   
    LOCATION = "\x1b[38;2;200;200;200m"
    EXTRA = "\x1b[94m" #"\x1b[38;2;140;140;140m"       
    EXCEPTION = "\x1b[38;2;255;100;100m"  
    MESSAGE = "\x1b[38;2;150;150;150m"  
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


class ReadableFormatterFile_v1(logging.Formatter):
    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    TIMESTAMP_C = "\x1b[38;2;200;200;200m"
    DEBUG_C = "\x1b[38;2;130;170;255m" 
    INFO_C = "\x1b[32;20m"
    WARNING_C = "\x1b[33;20m"  
    ERROR_C = "\x1b[38;2;255;80;80m"     
    CRITICAL_C = "\x1b[38;2;255;80;100m"   
    LOCATION_C = "\x1b[38;2;200;200;200m"
    EXCEPTION_C = "\x1b[38;2;255;100;100m"  
    PATH_C = "\x1b[38;2;120;200;255m"   
    AT_C = "\x1b[38;2;160;160;160m"
    EXTRA_C = "\x1b[38;2;140;140;140m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: DEBUG_C,
        logging.INFO: INFO_C,
        logging.WARNING: WARNING_C,
        logging.ERROR: ERROR_C,
        logging.CRITICAL: CRITICAL_C,
    }

    def __init__(self, *, use_color: bool = True, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.use_color = use_color
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
        
        c = self.LEVEL_COLORS.get(record.levelno, self.DEBUG_C) if self.use_color else ""
        ts_c = self.TIMESTAMP_C if self.use_color else ""
        loc_c = self.LOCATION_C if self.use_color else ""
        at_c = self.AT_C if self.use_color else ""
        path_c = self.PATH_C if self.use_color else ""
        ex_c = self.EXCEPTION_C if self.use_color else ""
        ext_c = self.EXTRA_C if self.use_color else ""
        r = self.RESET if self.use_color else ""

        header = f"{ts_c}{data.get('timestamp')}{r} | {c}{record.levelname:<5}{r}"

        location = (
            f"{loc_c}[{record.funcName}(){r}"
            f"{at_c} @ {r}"
            f"{loc_c}{record.filename}{r}"
            f"{at_c}:{r}"
            f"{loc_c}{record.lineno}]{r}"
        )

        message = data.get("message")
        
        path_str = ""
        if "path" in data:
            path_val = data.pop("path")
            path_str = f" {path_c}{path_val}{r}"

        used_keys = {"message", "timestamp", "exc_info", "stack_info"}
        extras = {k: v for k, v in data.items() if k not in used_keys}
        extra_str = f" {ext_c}>> {extras}{r}" if extras else ""

        formatted = f"{header} | {location} {message}{path_str}{extra_str}"

        if "exc_info" in data:
            formatted += f"\n{ex_c}{data['exc_info']}{r}"
            
        if "stack_info" in data:
            formatted += f"\n{data['stack_info']}"

        return formatted


class ReadableColorFormatter_tester(logging.Formatter):
    """log formatter."""

    _BUILTIN_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    TIMESTAMP = "\x1b[38;2;100;110;130m"
    DEBUG_COLOR = "\x1b[38;2;130;170;255m"
    INFO_COLOR = "\x1b[38;2;120;220;180m"
    WARNING_COLOR = "\x1b[38;2;255;200;100m"
    ERROR_COLOR = "\x1b[38;2;255;120;130m"
    CRITICAL_COLOR = "\x1b[38;2;255;80;100m"

    # location parts
    FUNC_COLOR = "\x1b[38;2;180;220;255m"   
    FILE_COLOR = "\x1b[38;2;255;180;120m"   
    LINE_COLOR = "\x1b[38;2;140;200;140m"   
    AT_COLOR = "\x1b[38;2;160;160;160m"     

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
