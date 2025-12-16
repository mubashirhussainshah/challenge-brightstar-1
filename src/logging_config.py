import logging
import sys
from pathlib import Path
from typing import Optional
import warnings
import transformers


class LoggingConfig:
    """Centralized logging configuration for the entire project"""

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    @staticmethod
    def setup_logging(
        level: str = "INFO",
        log_file: Optional[str] = None,
        use_colors: bool = True,
        suppress_warnings: bool = True,
        format_string: Optional[str] = None,
    ):
        """
        Setup centralized logging configuration for the entire application.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file. If None, only console logging is used.
            use_colors: Whether to use colored output in console (has no effect on file logging)
            suppress_warnings: Whether to suppress common library warnings
            format_string: Custom format string. If None, uses default format.
        """
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Default format string
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Setup handlers list
        handlers = []

        # Console handler with optional colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if use_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(format_string)
        else:
            console_formatter = logging.Formatter(format_string)

        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format=format_string,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

        # Suppress common warnings if requested
        if suppress_warnings:
            LoggingConfig._suppress_common_warnings()

        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging configured: level={level}, file={'enabled' if log_file else 'disabled'}"
        )

    @staticmethod
    def _suppress_common_warnings():
        """Suppress common noisy warnings from dependencies"""
        # General Python warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")

        # Transformers library
        transformers.logging.set_verbosity_error()

        # Optuna (if used)
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            pass

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name.

        Args:
            name: Logger name (typically __name__ of the module)

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output"""

    def format(self, record):
        # Save original levelname
        levelname = record.levelname

        # Add color codes
        if levelname in LoggingConfig.COLORS:
            record.levelname = (
                f"{LoggingConfig.COLORS[levelname]}{levelname}"
                f"{LoggingConfig.COLORS['RESET']}"
            )

        # Format the message
        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname

        return result


class ProgressLogger:
    """Helper class for logging progress in long-running operations"""

    def __init__(
        self, logger: logging.Logger, total: int, description: str = "Processing"
    ):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.last_percent = -1

    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        percent = int((self.current / self.total) * 100)

        # Log every 10% or at completion
        if percent >= self.last_percent + 10 or self.current >= self.total:
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} ({percent}%)"
            )
            self.last_percent = percent

    def complete(self):
        """Mark as complete"""
        self.logger.info(f"{self.description}: Complete ({self.total}/{self.total})")


# Convenience function for quick setup
def setup_project_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    module_name: str = "brightstar",
):
    """
    Quick setup function for project logging.

    Args:
        log_level: Logging level
        log_file: Optional log file path
        module_name: Name of the project/module for filtering
    """
    LoggingConfig.setup_logging(
        level=log_level, log_file=log_file, use_colors=True, suppress_warnings=True
    )

    return LoggingConfig.get_logger(module_name)
