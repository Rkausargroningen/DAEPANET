"""core/scenario.py – Switching signals and valve scenarios for DAE models.

Classes
-------
ValveId
    Identifies a valve at either the start or end of a pipe (e.g. 'P2:start').
SwitchEvent
    Represents a discrete valve open/close transition at time t.
Mode
    Active operating interval [t_start, t_end) with stationary valve states.
Scenario
    Full scenario specification with initial states, scheduled events, and total time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass(frozen=True)
class ValveId:
    """Identifies a valve at either the start or end of a pipe.

    Parameters
    ----------
    pipe_id : str
        ID of the pipe hosting the valve (e.g. 'P2').
    end : str
        Which end of the pipe: 'start' or 'end'.
    """

    pipe_id: str
    end: str

    def __post_init__(self) -> None:
        if self.end not in {"start", "end"}:
            raise ValueError(
                f"Valve end must be 'start' or 'end', got {self.end!r}"
            )
        if not isinstance(self.pipe_id, str) or not self.pipe_id:
            raise ValueError(f"pipe_id must be a non-empty string, got {self.pipe_id!r}")

    def __str__(self) -> str:
        return f"{self.pipe_id}:{self.end}"

    @classmethod
    def from_str(cls, s: str) -> ValveId:
        """Parse string form like 'P2:start' into ValveId."""
        if not isinstance(s, str) or ":" not in s:
            raise ValueError(f"Invalid valve string representation: {s!r}, expected 'PIPE_ID:END'")
        parts = s.split(":", 1)
        return cls(pipe_id=parts[0], end=parts[1])


@dataclass(frozen=True)
class SwitchEvent:
    """A switching event at a specific timestamp.

    Parameters
    ----------
    time : float
        Switching time in seconds (strictly between 0 and t_end).
    valve : ValveId
        The valve being switched.
    state : int
        New state: 1 (open) or 0 (closed).
    """

    time: float
    valve: ValveId
    state: int

    def __post_init__(self) -> None:
        if isinstance(self.state, bool) or self.state not in (0, 1):
            raise ValueError(f"Switch event state must be 0 or 1, got {self.state!r}")
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError(f"Switch event time must be a non-negative finite number, got {self.time!r}")


@dataclass(frozen=True)
class Mode:
    """An active operating mode over a time interval [t_start, t_end).

    Parameters
    ----------
    t_start : float
        Interval start time in seconds.
    t_end : float
        Interval end time in seconds.
    state : dict[ValveId, int]
        Mapping of valves to binary state (1=open, 0=closed).
    """

    t_start: float
    t_end: float
    state: Dict[ValveId, int]


@dataclass
class Scenario:
    """A switching scenario with initial valve states and scheduled switch events.

    Parameters
    ----------
    title : str
        Descriptive title of the scenario.
    initial : dict[ValveId, int]
        Initial state of each valve at t = 0 (1 = open, 0 = closed).
    events : list[SwitchEvent]
        Scheduled valve switching events.
    t_end : float
        Total scenario duration in seconds (must be > 0).
    """

    title: str
    initial: Dict[ValveId, int]
    events: List[SwitchEvent] = field(default_factory=list)
    t_end: float = 120.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate scenario properties, raising ValueError with plain-language explanations."""
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Scenario title must be a non-empty string.")

        if not math.isfinite(self.t_end) or self.t_end <= 0:
            raise ValueError(f"Scenario end time t_end must be > 0, got {self.t_end!r}.")

        # Validate initial states
        if not isinstance(self.initial, dict):
            raise ValueError("Scenario initial must be a dict mapping ValveId to state (0 or 1).")

        normalized_initial: Dict[ValveId, int] = {}
        for k, v in self.initial.items():
            valve_id = k if isinstance(k, ValveId) else ValveId.from_str(str(k))
            if isinstance(v, bool) or v not in (0, 1):
                raise ValueError(f"Initial valve state for {valve_id} must be 0 or 1, got {v!r}.")
            normalized_initial[valve_id] = int(v)
        self.initial = normalized_initial

        # Validate and normalize events
        normalized_events: List[SwitchEvent] = []
        for ev in self.events:
            if not isinstance(ev, SwitchEvent):
                raise ValueError(f"Expected SwitchEvent instance, got {ev!r}.")
            if ev.time <= 0 or ev.time >= self.t_end:
                raise ValueError(
                    f"Switch event time {ev.time}s must be strictly inside (0, t_end) = (0, {self.t_end}s)."
                )
            if ev.valve not in self.initial:
                raise ValueError(
                    f"Valve {ev.valve} in switch event has no initial state defined in initial states."
                )
            normalized_events.append(ev)

        # Sort events by time
        self.events = sorted(normalized_events, key=lambda e: e.time)

    def modes(self) -> List[Mode]:
        """Compute the sequence of operating modes across [0, t_end].

        Events occurring at the exact same timestamp are grouped into a single switch.
        Events that do not change the active state of a valve are dropped.

        Returns
        -------
        list of Mode
            Interval modes with start/end timestamps and valve state dictionaries.
        """
        curr_state = dict(self.initial)
        modes_list: List[Mode] = []
        prev_t = 0.0

        # Group sorted events by timestamp
        groups: Dict[float, List[SwitchEvent]] = {}
        for ev in self.events:
            groups.setdefault(ev.time, []).append(ev)

        for t_sw in sorted(groups):
            new_state = dict(curr_state)
            for ev in groups[t_sw]:
                new_state[ev.valve] = ev.state

            # If the group actually changes the state
            if new_state != curr_state:
                if t_sw > prev_t:
                    modes_list.append(Mode(t_start=prev_t, t_end=t_sw, state=dict(curr_state)))
                curr_state = new_state
                prev_t = t_sw

        # Final interval up to t_end
        modes_list.append(Mode(t_start=prev_t, t_end=self.t_end, state=dict(curr_state)))
        return modes_list

    def to_dict(self) -> dict:
        """Serialize scenario to JSON-compatible dict."""
        return {
            "title": self.title,
            "initial": {str(v): s for v, s in self.initial.items()},
            "events": [
                {"time": float(e.time), "valve": str(e.valve), "state": int(e.state)}
                for e in self.events
            ],
            "t_end": float(self.t_end),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        """Deserialize scenario from dict."""
        if not isinstance(data, dict):
            raise ValueError("Scenario data must be a dictionary.")

        title = str(data.get("title", ""))
        t_end = float(data.get("t_end", 120.0))

        raw_initial = data.get("initial", {})
        initial: Dict[ValveId, int] = {}
        for k, v in raw_initial.items():
            initial[ValveId.from_str(str(k))] = int(v)

        raw_events = data.get("events", [])
        events: List[SwitchEvent] = []
        for ed in raw_events:
            valve_val = ed["valve"]
            valve_id = valve_val if isinstance(valve_val, ValveId) else ValveId.from_str(str(valve_val))
            events.append(
                SwitchEvent(
                    time=float(ed["time"]),
                    valve=valve_id,
                    state=int(ed["state"]),
                )
            )

        return cls(title=title, initial=initial, events=events, t_end=t_end)

    @classmethod
    def matlab_specimen(cls, k: int, v1: ValveId, v2: ValveId) -> Scenario:
        """Factory for MATLAB reference specimen scenarios (k = 1..5)."""
        titles = {
            1: "valve 1 closed at t_s = 60 s (started from rest)",
            2: "no switching: relaxation from rest onto EPANET's steady state",
            3: "equilibrium first, then valve 1 closes at t_s = 300 s",
            4: "equilibrium first, then valve 2 closes at t_s = 300 s",
            5: "valve 1 opened at t_s = 60 s (start with valve 1 closed)",
        }
        if k not in titles:
            raise ValueError(f"k must be in 1..5, got {k}")

        if k == 1:
            return cls(
                title=titles[1],
                initial={v1: 1, v2: 1},
                events=[SwitchEvent(time=60.0, valve=v1, state=0)],
                t_end=120.0,
            )
        elif k == 2:
            return cls(
                title=titles[2],
                initial={v1: 1, v2: 1},
                events=[],
                t_end=200.0,
            )
        elif k == 3:
            return cls(
                title=titles[3],
                initial={v1: 1, v2: 1},
                events=[SwitchEvent(time=300.0, valve=v1, state=0)],
                t_end=400.0,
            )
        elif k == 4:
            return cls(
                title=titles[4],
                initial={v1: 1, v2: 1},
                events=[SwitchEvent(time=300.0, valve=v2, state=0)],
                t_end=400.0,
            )
        elif k == 5:
            return cls(
                title=titles[5],
                initial={v1: 0, v2: 1},
                events=[SwitchEvent(time=60.0, valve=v1, state=1)],
                t_end=300.0,
            )
