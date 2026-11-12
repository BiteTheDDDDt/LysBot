"""Owned execution scopes for capabilities and scheduled workflows."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Callable
from .network_retry import retry_scope

from .conversation import (
    activity_scope, current_scope, ensure_current_session, ensure_send_allowed, session_scope,
)


@dataclass(frozen=True)
class ExecutionPolicy:
    name: str
    permission: str = "current"  # current session or permission to send in it
    timeout: Callable[[], float] | None = None
    slots: Callable[[], object] | None = None
    check_result: bool = True


class Resources(AsyncExitStack):
    async def __aexit__(self, *exception):
        # Shutdown can cancel both a parent and its awaited child. Finish owned
        # cleanup even if that delivers a second cancellation, then propagate it.
        task = asyncio.create_task(super().__aexit__(*exception))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    def own_client(self, client):
        self.push_async_callback(client.aclose)
        return client

    async def create(self, factory, cleanup, *args):
        """A thread cannot be cancelled: collect its result before releasing it."""
        task = asyncio.create_task(asyncio.to_thread(factory, *args))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                result = task.result()
            except (Exception, asyncio.CancelledError):
                pass
            else:
                cleanup(result)
            raise
        self.callback(cleanup, result)
        return result


@asynccontextmanager
async def execution_scope(policy, *, session_id=None, run_id=None, timeout=None, slots=None):
    """Check both sides of queueing/execution; cleanup outlives the work timeout."""
    state = current_scope()
    sid = session_id or (state.session_id if state else None)
    if sid is None:
        # Model/command callers can carry scope without a serialized chat turn.
        from .model_types import current_session_id
        sid = current_session_id.get()
    if sid == "_unknown_":
        sid = None
    group_id = int(sid[6:]) if sid and sid.startswith("group_") else None

    def check():
        if policy.permission == "send":
            ensure_send_allowed(group_id)
        elif policy.permission == "current":
            ensure_current_session()
        else:
            raise ValueError(f"Unknown execution permission: {policy.permission}")

    with session_scope(sid) if sid else nullcontext():
        with activity_scope(policy.name, kind="operation", session_id=sid, run_id=run_id) as activity:
            async with Resources() as resources:
                limit = timeout if timeout is not None else (policy.timeout() if policy.timeout else None)
                gate = slots if slots is not None else (policy.slots() if policy.slots else None)
                try:
                    async with asyncio.timeout(limit), AsyncExitStack() as retry_stack:
                        if limit is not None:
                            retry_stack.enter_context(retry_scope(limit))
                        check()
                        activity["phase"] = "queued" if gate else "running"
                        async with gate if gate is not None else AsyncExitStack():
                            check()
                            activity["phase"] = "running"
                            yield resources
                            if policy.check_result:
                                check()
                finally:
                    activity["phase"] = "cleaning"


def operation(policy):
    """Declare lifecycle at a reusable service entry point (no tool registration)."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            async with execution_scope(policy, session_id=kwargs.get("session_id")):
                return await function(*args, **kwargs)
        return wrapped
    return decorate
