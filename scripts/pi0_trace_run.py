#!/usr/bin/env python3
"""Trace SOFollower action dispatch during a normal ``lerobot-rollout``.

This wrapper preserves the real hardware writes while periodically logging the
requested joint targets, measured positions, servo goal registers, and torque
state.  It is intended for short, supervised deployment diagnostics.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from lerobot.robots.so_follower.so_follower import SOFollower


LOG_PATH = Path(
    os.environ.get("LEROBOT_TRACE_LOG", str(Path(tempfile.gettempdir()) / "lerobot_pi0_trace.jsonl"))
)
PRINT_EVERY = max(1, int(os.environ.get("LEROBOT_TRACE_PRINT_EVERY", "30")))

_original_connect = SOFollower.connect
_original_disconnect = SOFollower.disconnect
_original_get_observation = SOFollower.get_observation
_original_send_action = SOFollower.send_action
_step = 0


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _read_register(bus: Any, name: str) -> dict[str, Any] | str:
    try:
        return bus.sync_read(name, normalize=False, num_retry=10)
    except Exception as exc:  # Diagnostics must not interrupt safe teardown.
        return f"{type(exc).__name__}: {exc}"


def traced_connect(self: SOFollower, *args: Any, **kwargs: Any) -> None:
    _original_connect(self, *args, **kwargs)
    torque = _read_register(self.bus, "Torque_Enable")
    print(f"[TRACE] Torque after connect: {torque}", flush=True)


def traced_get_observation(self: SOFollower) -> dict[str, Any]:
    observation = _original_get_observation(self)
    self._trace_last_observation = observation
    return observation


def traced_send_action(self: SOFollower, action: dict[str, Any]) -> dict[str, Any]:
    global _step
    _step += 1

    actual = {
        key: _as_float(value)
        for key, value in getattr(self, "_trace_last_observation", {}).items()
        if key.endswith(".pos")
    }
    requested = {key: _as_float(value) for key, value in action.items() if key.endswith(".pos")}
    sent = _original_send_action(self, action)

    if _step == 1 or _step % PRINT_EVERY == 0:
        torque = _read_register(self.bus, "Torque_Enable")
        goal_raw = _read_register(self.bus, "Goal_Position")
        present_raw = _read_register(self.bus, "Present_Position")
        delta = {key: requested[key] - actual[key] for key in requested if key in actual}
        row = {
            "timestamp": time.time(),
            "step": _step,
            "requested": requested,
            "actual": actual,
            "delta": delta,
            "sent": {key: _as_float(value) for key, value in sent.items()},
            "torque_enable": torque,
            "goal_position_raw": goal_raw,
            "present_position_raw": present_raw,
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        compact = {key.removesuffix(".pos"): round(value, 2) for key, value in delta.items()}
        print(f"[TRACE] step={_step} delta={compact} torque={torque}", flush=True)

    return sent


def traced_disconnect(self: SOFollower) -> None:
    print(f"[TRACE] Total action writes before disconnect: {_step}", flush=True)
    _original_disconnect(self)


SOFollower.connect = traced_connect
SOFollower.get_observation = traced_get_observation
SOFollower.send_action = traced_send_action
SOFollower.disconnect = traced_disconnect


if __name__ == "__main__":
    from lerobot.scripts.lerobot_rollout import main

    print(f"[TRACE] Hardware action trace: {LOG_PATH}", flush=True)
    main()
