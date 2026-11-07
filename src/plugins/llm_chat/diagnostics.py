"""Small, credential-free API failure records for model context."""
import re
import time


def api_failure(error, *, operation):
    # Never copy arbitrary response bodies, headers, request URLs or tracebacks.
    chain, seen, current = [], set(), error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    text = " ".join(f"{type(item).__name__} {item}" for item in chain).lower()
    status = next((value for item in chain
                   if isinstance(value := (getattr(item, "status_code", None)
                       or getattr(getattr(item, "response", None), "status_code", None)), int)), None)
    if not isinstance(status, int):
        match = re.search(r"(?:error code:|http)\s*(\d{3})", text)
        status = int(match[1]) if match else None
    code, reason, retryable = "request_failed", "请求失败，未取得有效结果", False
    for markers, values in (
        (("usage_limit_reached", "insufficient_quota", "arrearage", "balance is insufficient"),
         ("quota_exhausted", "服务端额度或余额不足，立即重复请求通常无效", False)),
        (("invalid_api_key", "unauthorized", "authentication"),
         ("authentication_failed", "服务端拒绝身份认证", False)),
        (("timeout", "timed out"), ("timeout", "请求超时，不能据此判断没有资料", True)),
        (("rate limit", "rate_limit", "model_cooldown"),
         ("rate_limited", "服务端限流或模型冷却", True)),
        (("connection", "connecterror"), ("connection_failed", "连接上游服务失败", True)),
    ):
        if any(marker in text or marker in type(error).__name__.lower() for marker in markers):
            code, reason, retryable = values
            break
    else:
        if status == 429:
            code, reason, retryable = "rate_limited", "服务端限流", True
        elif status in {401, 403}:
            code, reason = "access_denied", "服务端拒绝访问"
        elif status and status >= 500:
            code, reason, retryable = "upstream_unavailable", "上游服务异常", True
        elif status == 400:
            code, reason = "invalid_request", "服务端拒绝请求参数，需要修正请求"
    result = {"operation": operation, "code": code, "message": reason,
              "http_status": status, "retryable": retryable, "observed_at": time.time()}
    delay = re.search(r"(?:reset_seconds|retry_after)[\"']?\s*[:=]\s*(\d+)", text)
    if delay:
        result["retry_after_seconds"] = min(int(delay[1]), 604800)
    return result


def error_summary(error):
    """Give empty timeout/cancellation exceptions a useful, stable log message."""
    if error is None:
        return "无错误详情"
    detail = str(error).strip()
    if (not detail or detail.endswith(":")) and error.__cause__ is not None:
        detail = (detail + " " + error_summary(error.__cause__)).strip()
    if not detail:
        detail = {"TimeoutError": "请求达到时间限制", "CancelledError": "调用被上层取消（会话取消、停机或总预算耗尽）"}.get(type(error).__name__, "无错误详情")
    return f"{type(error).__name__}: {detail}"
