"""Serialize conversation turns and invalidate pending sends after session reset."""
import asyncio
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Optional


class TurnSuperseded(asyncio.CancelledError):
    """A control command invalidated this turn before its next side effect."""


@dataclass
class ActiveTurn:
    session_id: str
    generation: int
    owner: asyncio.Task
    serialized: bool = False


_locks: dict[str, asyncio.Lock] = {}
_users: dict[str, int] = {}
_generations: dict[str, int] = {}
_forgotten_before: dict[str, int] = {}
_active_tasks: set[asyncio.Task] = set()
_current: ContextVar[Optional[ActiveTurn]] = ContextVar("conversation_turn", default=None)
_closing = False
_activities: dict[str, dict] = {}
_activity_parent: ContextVar[Optional[str]] = ContextVar("activity_parent", default=None)


@contextmanager
def activity_scope(name: str, *, kind: str, run_id: Optional[str] = None,
                   session_id: Optional[str] = None):
    """Observe existing work without spawning tasks or changing cancellation rules.

    Each nested activity owns its record; a copied context never lets a child
    remove its parent's record. Inputs and generated content are not recorded.
    """
    owner = asyncio.current_task()
    state = _current.get()
    key = uuid.uuid4().hex
    _activities[key] = {
        "id": key, "parent_id": _activity_parent.get(), "name": name, "kind": kind,
        "session_id": state.session_id if state else session_id,
        "generation": state.generation if state else (current_generation(session_id) if session_id else None),
        "run_id": run_id, "started_at": time.monotonic(), "owner": owner,
        "phase": "running",
    }
    token = _activity_parent.set(key)
    already_tracked = owner in _active_tasks
    _active_tasks.add(owner)
    try:
        yield _activities[key]
    finally:
        _activity_parent.reset(token)
        _activities.pop(key, None)
        if not already_tracked:
            _active_tasks.discard(owner)


def task_snapshot() -> dict:
    """Compact local diagnostics; several activities can belong to one Task."""
    now = time.monotonic()
    activities = []
    for record in list(_activities.values())[:100]:
        row = {key: value for key, value in record.items() if key not in {"owner", "started_at"}}
        row["elapsed_seconds"] = round(now - record["started_at"], 1)
        row["cancelling"] = bool(record["owner"] and record["owner"].cancelling())
        activities.append(row)
    return {"task_count": sum(not task.done() for task in _active_tasks if task),
            "activity_count": len(_activities), "activities": activities}


def tracked_job(function):
    """Drain scheduled producers before their database and HTTP clients close."""
    @wraps(function)
    async def wrapped(*args, **kwargs):
        if _closing:
            return
        task = asyncio.current_task()
        already_tracked = task in _active_tasks
        _active_tasks.add(task)
        try:
            with activity_scope(f"{function.__module__.rsplit('.', 1)[-1]}.{function.__name__}", kind="scheduled"):
                return await function(*args, **kwargs)
        finally:
            if not already_tracked:
                _active_tasks.discard(task)
    return wrapped


def session_key(user_id: int, group_id: Optional[int] = None) -> str:
    return f"group_{group_id}" if group_id else f"user_{user_id}"


def invalidate(session_id: str, *, forget: bool = False) -> None:
    _generations[session_id] = _generations.get(session_id, 0) + 1
    if forget:
        _forgotten_before[session_id] = _generations[session_id]


def current_generation(session_id: str) -> int:
    """Read the live generation when accepting a new event or queued batch."""
    return _generations.get(session_id, 0)


def current_scope():
    return _current.get()


@contextmanager
def session_scope(session_id: str, *, generation: Optional[int] = None):
    """Capture a session generation without locking out ordinary chat turns.

    Incoming enrichment and scheduled work inherit the original generation across
    awaits. A nested scope must not refresh an obsolete job after a reset.
    """
    owner = asyncio.current_task()
    previous = _current.get()
    if generation is None:
        generation = (previous.generation if previous and previous.session_id == session_id
                      else current_generation(session_id))
    serialized = bool(previous and previous.session_id == session_id
                      and previous.generation == generation
                      and previous.owner is owner and previous.serialized)
    token = _current.set(ActiveTurn(session_id, generation, owner, serialized))
    already_tracked = owner in _active_tasks
    _active_tasks.add(owner)
    try:
        with activity_scope(session_id, kind="session"):
            yield
    finally:
        _current.reset(token)
        if not already_tracked:
            _active_tasks.discard(owner)


def ensure_current_session() -> None:
    """Reject obsolete work without preventing incoming records."""
    state = _current.get()
    if _closing or (state and state.generation != _generations.get(state.session_id, 0)):
        raise TurnSuperseded()


def memory_write_allowed(session_id: str) -> bool:
    state = _current.get()
    return not (state and state.session_id == session_id
                and state.generation < _forgotten_before.get(session_id, 0))


def ensure_send_allowed(group_id: Optional[int] = None) -> None:
    ensure_current_session()


def serialized_turn(key_fn, *, background: bool = False):
    """Capture the generation before queueing, so queued work cannot survive reset."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            sid = key_fn(*args, **kwargs)
            state = _current.get()
            owner = asyncio.current_task()
            if state and state.owner is owner and state.session_id == sid and state.serialized:
                return await function(*args, **kwargs)
            generation = (state.generation if state and state.session_id == sid
                          else _generations.get(sid, 0))
            lock = _locks.setdefault(sid, asyncio.Lock())
            if background and lock.locked():
                return False
            _users[sid] = _users.get(sid, 0) + 1
            already_tracked = owner in _active_tasks
            _active_tasks.add(owner)
            try:
                async with lock:
                    if _closing or generation != _generations.get(sid, 0):
                        return
                    token = _current.set(ActiveTurn(sid, generation, owner, serialized=True))
                    try:
                        group_id = int(sid[6:]) if sid.startswith("group_") else None
                        ensure_send_allowed(group_id)
                        with activity_scope(function.__name__, kind="conversation"):
                            return await function(*args, **kwargs)
                    except TurnSuperseded:
                        return
                    finally:
                        _current.reset(token)
            finally:
                if not already_tracked:
                    _active_tasks.discard(owner)
                _users[sid] -= 1
                if not _users[sid]:
                    _users.pop(sid, None)
                    _locks.pop(sid, None)
        return wrapped
    return decorate


def start() -> None:
    global _closing
    _closing = False


async def close() -> None:
    global _closing
    _closing = True
    tasks = [t for t in _active_tasks if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
