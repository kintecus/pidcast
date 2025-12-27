"""Retry decorator and error classification for evals."""

import functools
import logging
import time
from typing import Callable, Type

from groq import APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

# Retryable error types (transient errors that should trigger retry)
RETRYABLE_ERRORS: tuple[Type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

# Non-retryable error types (permanent errors that should fail immediately)
# These will be caught as generic exceptions and won't retry
NON_RETRYABLE_ERROR_MESSAGES = [
    "invalid_api_key",
    "invalid_request_error",
    "authentication",
    "unauthorized",
    "bad request",
]


def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exponential_base: float = 2.0,
):
    """
    Decorator to retry function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds (default 2.0)
        exponential_base: Base for exponential backoff (default 2.0)

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(max_retries=3, base_delay=2.0)
        def api_call():
            # This will retry up to 3 times with delays of 2s, 4s, 8s
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            retry_count = 0

            for attempt in range(max_retries + 1):
                try:
                    # Try to execute the function
                    result = func(*args, **kwargs)

                    # If successful and we retried, log it
                    if retry_count > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded after {retry_count} "
                            f"{'retry' if retry_count == 1 else 'retries'}"
                        )

                    return result

                except RETRYABLE_ERRORS as e:
                    last_exception = e
                    retry_count += 1

                    if attempt < max_retries:
                        delay = base_delay * (exponential_base**attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts"
                        )
                        raise

                except Exception as e:
                    # Check if it's a non-retryable error
                    error_str = str(e).lower()
                    if any(msg in error_str for msg in NON_RETRYABLE_ERROR_MESSAGES):
                        logger.error(
                            f"{func.__name__} failed with non-retryable error: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # Unknown error - don't retry to be safe
                    logger.error(
                        f"{func.__name__} failed with unexpected error: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise

            # This shouldn't be reached, but just in case
            if last_exception:
                raise last_exception

        # Store retry count on the wrapper for inspection
        wrapper._retry_count = 0

        return wrapper

    return decorator


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error should trigger a retry.

    Args:
        error: Exception instance

    Returns:
        True if error is retryable, False otherwise
    """
    # Check if it's a known retryable error type
    if isinstance(error, RETRYABLE_ERRORS):
        return True

    # Check if error message indicates non-retryable
    error_str = str(error).lower()
    if any(msg in error_str for msg in NON_RETRYABLE_ERROR_MESSAGES):
        return False

    # For unknown errors, don't retry (safer default)
    return False
