"""One bounded retry policy for outbound API requests, never whole tool turns."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time

from .config import config
from .logging_setup import general_logger

_deadline = ContextVar('network_retry_deadline', default=None)


class ResponseError(RuntimeError):
    def __init__(self, message, *, status_code, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class StepTimeout(TimeoutError):
    """A local per-step budget expired; the request may already be in flight.

    Raised instead of a bare TimeoutError so the retry policy cannot mistake a
    locally enforced deadline for a transient upstream failure (the cause chain
    still contains TimeoutError, hence the explicit check in retry_delay).
    """


def retry_budget(single_request_timeout):
    retries = max(0, min(2, int(config.network_max_retries)))
    return float(single_request_timeout) * (retries + 1) + max(0, config.network_retry_delay) * retries


def remaining_retry_budget():
    deadline = _deadline.get()
    return deadline - time.monotonic() if deadline is not None else float('inf')


@contextmanager
def retry_scope(seconds):
    old = _deadline.get()
    limit = time.monotonic() + max(0, seconds)
    token = _deadline.set(min(old, limit) if old is not None else limit)
    try:
        yield
    finally:
        _deadline.reset(token)


def retry_delay(error, *, replay_safe=True):
    from .diagnostics import api_failure
    if isinstance(error, StepTimeout):
        return None
    if api_failure(error, operation='request')['code'] in {'quota_exhausted', 'authentication_failed', 'access_denied', 'invalid_request'}:
        return None
    chain, seen, current = [], set(), error
    while current is not None and id(current) not in seen:
        seen.add(id(current)); chain.append(current); current = current.__cause__
    statuses = [getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None) for e in chain]
    status = next((value for value in statuses if isinstance(value, int)), None)
    names = {type(e).__name__ for e in chain}
    connect = bool(names & {'ConnectError', 'ConnectTimeout', 'ConnectionRefusedError'})
    transient = status in {429, 500, 502, 503, 504} or connect
    if replay_safe:
        transient |= bool(names & {'APIConnectionError', 'APITimeoutError', 'ReadTimeout', 'ReadError', 'RemoteProtocolError', 'ConnectionResetError', 'TimeoutError'})
    elif not connect and status != 429:
        return None  # A submitted image/send request may already have taken effect.
    if not transient:
        return None
    delay = float(config.network_retry_delay)
    for e in chain:
        response = getattr(e, 'response', None)
        headers = getattr(e, 'headers', None) or getattr(response, 'headers', {})
        value = headers.get('retry-after') or headers.get('Retry-After')
        if value is not None:
            try:
                delay = float(value)
            except (ValueError, TypeError):
                try:
                    delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    continue
            break
        body = getattr(e, 'body', None)
        detail = body.get('error', body) if isinstance(body, dict) else {}
        if detail.get('code') == 'model_cooldown' and isinstance(detail.get('reset_seconds'), (int, float)):
            delay = float(detail['reset_seconds']); break
    if not math.isfinite(delay) or delay > 60:
        return None
    return max(0, delay)


async def request_with_retry(request, *, operation, replay_safe=True, remaining=None, on_retry=None):
    from .conversation import ensure_current_session
    retries = max(0, min(2, int(config.network_max_retries)))
    for attempt in range(retries + 1):
        ensure_current_session()
        try:
            result = await request()
            if attempt:
                general_logger.info(f'[network_retry] operation={operation} outcome=recovered attempts={attempt + 1}')
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            delay = retry_delay(error, replay_safe=replay_safe)
            deadline = _deadline.get()
            left = deadline - time.monotonic() if deadline is not None else float('inf')
            if remaining is not None:
                left = min(left, remaining())
            if attempt >= retries or delay is None or left <= delay + 5:
                raise
            general_logger.warning(f'[network_retry] operation={operation} attempt={attempt + 2} delay={delay:g}s error={type(error).__name__}')
            if on_retry is not None:
                on_retry(error)
            await asyncio.sleep(delay)


async def post_with_retry(client, url, *, headers, body, operation, files=None, replay_safe=False, call=None):
    async def request():
        if call is not None:
            call.start()
        response = await client.post(url, headers=headers, **({'data': body, 'files': files} if files else {'json': body}))
        if response.status_code >= 400:
            raise ResponseError(f'HTTP {response.status_code}: {response.text[:300]}',
                                status_code=response.status_code, headers=response.headers)
        return response
    return await request_with_retry(request, operation=operation, replay_safe=replay_safe,
                                    on_retry=call.retry if call is not None else None)


@asynccontextmanager
async def stream_with_retry(client, url, *, headers, body, operation, call=None):
    """Retry connection/rejection only; never replay a stream already accepted."""
    async def connect():
        stack = AsyncExitStack()
        try:
            if call is not None:
                call.start()
            response = await stack.enter_async_context(client.stream('POST', url, headers=headers, json=body))
            if response.status_code >= 400:
                detail = (await response.aread()).decode('utf-8', errors='replace')[:300]
                raise ResponseError(f'HTTP {response.status_code}: {detail}',
                                    status_code=response.status_code, headers=response.headers)
            return stack, response
        except BaseException:
            await stack.aclose()
            raise
    stack, response = await request_with_retry(connect, operation=operation, replay_safe=False,
                                               on_retry=call.retry if call is not None else None)
    async with stack:
        yield response
