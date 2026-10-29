from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import colorlog
import openai
from pydantic import ValidationError

from memslides.utils.constants import LOGGING_LEVEL


P = ParamSpec("P")
R = TypeVar("R")

_context_logger: ContextVar[logging.Logger | None] = ContextVar(
    "memslides_context_logger",
    default=None,
)

_CONSOLE_FORMAT = (
    "%(log_color)s%(levelname)-4s%(reset)s %(asctime)s "
    "[%(name)s] %(blue)s%(message)s"
)
_FILE_FORMAT = "%(levelname)-4s %(asctime)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _file_handler_path(handler: logging.Handler) -> str | None:
    if not isinstance(handler, logging.FileHandler):
        return None
    filename = getattr(handler, "baseFilename", None)
    return str(Path(filename)) if filename else None


def _make_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(LOGGING_LEVEL)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            _CONSOLE_FORMAT,
            datefmt=_DATE_FORMAT,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    return handler


def _attach_file_handler(logger: logging.Logger, log_file: str | Path) -> None:
    path = Path(log_file)
    normalized = str(path)
    if any(_file_handler_path(handler) == normalized for handler in logger.handlers):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)


def create_logger(name: str = __name__, log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger without duplicating handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(_make_console_handler())

    if log_file is not None:
        _attach_file_handler(logger, log_file)

    return logger


def set_logger(name: str = __name__, log_file: str | Path | None = None) -> logging.Logger:
    """Set the logger for the current async context."""

    current = _context_logger.get()
    if current is not None and current.name != "default logger":
        raise AssertionError("Context logger is already set.")

    logger = create_logger(name, log_file)
    _context_logger.set(logger)
    logger.debug("Context logger initialized with loglevel=%s", LOGGING_LEVEL)
    return logger


def reset_context_logger() -> None:
    _context_logger.set(None)


@contextmanager
def isolated_context_logger():
    token = _context_logger.set(None)
    try:
        yield
    finally:
        _context_logger.reset(token)


def get_logger() -> logging.Logger:
    logger = _context_logger.get()
    if logger is None:
        logger = create_logger("default logger")
        _context_logger.set(logger)
    return logger


def debug(msg, *args, **kwargs) -> None:
    get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs) -> None:
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs) -> None:
    get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs) -> None:
    get_logger().error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs) -> None:
    get_logger().critical(msg, *args, **kwargs)


def exception(msg, *args, **kwargs) -> None:
    get_logger().exception(msg, *args, **kwargs)


class timer:
    """Timer context manager and decorator that logs slow operations."""

    def __init__(self, name: str | None = None, threshold_seconds: float = 1.0):
        self.name = name
        self.threshold_seconds = threshold_seconds
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if self.start_time is None:
            return
        self._log_elapsed(self.name or "operation", time.perf_counter() - self.start_time)

    def _log_elapsed(self, label: str, elapsed: float) -> None:
        if elapsed > self.threshold_seconds:
            debug("%s took %.2f seconds", label, elapsed)

    def __call__(
        self,
        func: Callable[P, R] | Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, R] | Callable[P, Coroutine[Any, Any, R]]:
        label = self.name or getattr(func, "__name__", "operation")

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    self._log_elapsed(label, time.perf_counter() - started)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._log_elapsed(label, time.perf_counter() - started)

        return sync_wrapper


_OPENAI_EXCEPTION_LABELS: tuple[tuple[type[BaseException], str], ...] = (
    (openai.RateLimitError, "RateLimitError (HTTP 429)"),
    (openai.APITimeoutError, "APITimeoutError"),
    (openai.APIConnectionError, "APIConnectionError"),
    (openai.AuthenticationError, "AuthenticationError (HTTP 401)"),
    (openai.PermissionDeniedError, "PermissionDeniedError (HTTP 403)"),
    (openai.NotFoundError, "NotFoundError (HTTP 404)"),
    (openai.ConflictError, "ConflictError (HTTP 409)"),
    (openai.BadRequestError, "BadRequestError (HTTP 400)"),
    (openai.UnprocessableEntityError, "UnprocessableEntityError (HTTP 422)"),
    (openai.InternalServerError, "InternalServerError (HTTP 500)"),
    (openai.APIResponseValidationError, "APIResponseValidationError"),
    (openai.InvalidWebhookSignatureError, "InvalidWebhookSignatureError"),
    (openai.ContentFilterFinishReasonError, "ContentFilterFinishReasonError"),
    (openai.LengthFinishReasonError, "LengthFinishReasonError"),
    (openai.APIError, "APIError"),
    (openai.OpenAIError, "OpenAIError"),
    (ValidationError, "Pydantic ValidationError"),
)


def _exception_message(exc: Exception) -> str:
    for exc_type, label in _OPENAI_EXCEPTION_LABELS:
        if isinstance(exc, exc_type):
            return f"{label}: {exc}"

    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "status_code", "unknown")
        return f"APIStatusError (HTTP {code}): {exc}"

    if hasattr(exc, "http_status"):
        return f"OpenAI API Error {exc.http_status}: {exc}"

    return f"Exception: {exc}\n{traceback.format_exc()}"


def logging_openai_exceptions(identifider: str | Any, exc: Exception) -> str:
    msg = _exception_message(exc)
    warning("%s encountered %s", identifider, msg)
    return msg
