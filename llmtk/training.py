"""Training-loop instrumentation: throughput measurement and divergence detection."""

import math
import statistics
import time
from collections import deque


class ThroughputMeter:
    """Tokens/s measured over windows of `window` steps, ignoring the first `warmup_steps` steps.

    Call `step()` once per optimizer step. Before a step for which `should_sync()` is True, the
    caller must synchronize the device so wall time reflects finished GPU work. Call
    `reset_window()` (after a sync) when doing untimed work such as evaluation or checkpointing,
    so it is not counted.
    """

    def __init__(self, tokens_per_step: int, window: int = 50, warmup_steps: int = 20) -> None:
        if warmup_steps < 1 or window < 1:
            raise ValueError("warmup_steps and window must be >= 1")
        self.tokens_per_step = tokens_per_step
        self.window = window
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.rates: list[float] = []
        self._n = 0  # steps in the current window
        self._t0: float | None = None

    def should_sync(self) -> bool:
        """True when the next `step()` call ends warmup or closes a window."""
        nxt = self.steps + 1
        return nxt == self.warmup_steps or (nxt > self.warmup_steps and self._n + 1 == self.window)

    def step(self) -> float | None:
        """Record one step; returns the window's tokens/s when a window closes, else None."""
        self.steps += 1
        if self.steps < self.warmup_steps:
            return None
        if self.steps == self.warmup_steps:
            self.reset_window()
            return None
        self._n += 1
        if self._n == self.window:
            now = time.perf_counter()
            rate = self.window * self.tokens_per_step / (now - self._t0)
            self.rates.append(rate)
            self._t0, self._n = now, 0
            return rate
        return None

    def reset_window(self) -> None:
        """Discard the partial window and restart timing now (no-op during warmup)."""
        if self.steps >= self.warmup_steps:
            self._t0, self._n = time.perf_counter(), 0

    @property
    def median(self) -> float | None:
        """Median tokens/s over all closed windows."""
        return statistics.median(self.rates) if self.rates else None


class DivergenceError(RuntimeError):
    """Raised when the training loss is non-finite or spikes far above its recent median."""


class DivergenceDetector:
    """Flags NaN/inf losses and losses above `k` x the running median of the last `window` losses.

    The spike check starts once `min_history` losses have been seen; `k <= 0` disables it.
    """

    def __init__(self, k: float = 3.0, window: int = 100, min_history: int = 50) -> None:
        self.k = k
        self.history: deque[float] = deque(maxlen=window)
        self.min_history = min_history

    def check(self, step: int, loss: float) -> None:
        """Raise DivergenceError if `loss` at `step` looks divergent; otherwise record it."""
        if not math.isfinite(loss):
            raise DivergenceError(f"step {step}: loss is {loss} (non-finite); stopping.")
        if self.k > 0 and len(self.history) >= self.min_history:
            med = statistics.median(self.history)
            if loss > self.k * med:
                raise DivergenceError(
                    f"step {step}: loss {loss:.4f} > {self.k:g} x running median {med:.4f} "
                    f"(last {len(self.history)} steps); stopping."
                )
        self.history.append(loss)
