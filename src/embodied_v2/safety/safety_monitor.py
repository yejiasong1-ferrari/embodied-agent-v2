"""Runtime payload-stability policy with hard local stop authority."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStabilitySample:
    slip_m: float
    tilt_deg: float
    contact_persisted: bool
    peak_force_n: float
    instability_increasing: bool = False


class RuntimeSafetyMonitor:
    def __init__(
        self,
        *,
        hard_force_n: float = 12.0,
        warning_slip_m: float = 0.003,
        stop_slip_m: float = 0.012,
        stop_tilt_deg: float = 12.0,
    ):
        self.hard_force_n = float(hard_force_n)
        self.warning_slip_m = float(warning_slip_m)
        self.stop_slip_m = float(stop_slip_m)
        self.stop_tilt_deg = float(stop_tilt_deg)

    def evaluate(self, sample: RuntimeStabilitySample) -> str:
        if not sample.contact_persisted:
            return "stop_contact_lost"
        if sample.slip_m >= self.stop_slip_m or sample.tilt_deg >= self.stop_tilt_deg:
            return "stop_runtime_instability"
        if sample.peak_force_n >= self.hard_force_n and sample.instability_increasing:
            return "stop_force_and_instability"
        if sample.peak_force_n >= self.hard_force_n:
            return "hold_force_continue_slowly"
        if sample.slip_m >= self.warning_slip_m:
            return "bounded_preload_and_slow_down"
        return "stable_hold"
