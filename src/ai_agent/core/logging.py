"""Structured logging setup."""

import structlog
from ai_agent.core.config import get_settings


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if get_settings().log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(get_settings().log_level)
        ),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
