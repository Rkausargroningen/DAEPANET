"""Data models for water distribution networks.

Classes
-------
Node
    Represents a network node (Junction, Tank, Reservoir).
Pipe
    Represents a directed connection between two nodes, optionally containing a pump.
Network
    Container for nodes and pipes with validation and ID management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_finite_number(name: str, val: Any) -> Optional[float]:
    """Check that val is None or a finite real number (not bool, nan, inf)."""
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a number or None, got {val!r}")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    return f_val


def _validate_node_fields(
    node_type: str,
    capacity: Any,
    elevation: Any,
    head: Any,
) -> None:
    c = _check_finite_number("capacity", capacity)
    if c is not None and c < 0:
        raise ValueError(f"capacity must be >= 0, got {c}")

    _check_finite_number("elevation", elevation)

    h = _check_finite_number("head", head)
    if h is not None and node_type != "reservoir":
        raise ValueError(f"head is only allowed on reservoirs, not on {node_type!r}")


def _validate_pipe_fields(
    length: Any,
    diameter: Any,
    roughness: Any,
    pump_power: Any = 0.0,
    has_pump: Any = False,
    valve_start: Any = False,
    valve_end: Any = False,
) -> None:
    l_ = _check_finite_number("length", length)
    if l_ is not None and l_ <= 0:
        raise ValueError(f"length must be > 0, got {l_}")

    d = _check_finite_number("diameter", diameter)
    if d is not None and d <= 0:
        raise ValueError(f"diameter must be > 0, got {d}")

    r = _check_finite_number("roughness", roughness)
    if r is not None and r < 0:
        raise ValueError(f"roughness must be >= 0, got {r}")

    p = _check_finite_number("pump_power", pump_power)
    if p is not None and p < 0:
        raise ValueError(f"pump_power must be >= 0, got {p}")

    if not isinstance(has_pump, bool):
        raise ValueError(f"has_pump must be a boolean, got {has_pump!r}")

    if not isinstance(valve_start, bool):
        raise ValueError(f"valve_start must be a boolean, got {valve_start!r}")

    if not isinstance(valve_end, bool):
        raise ValueError(f"valve_end must be a boolean, got {valve_end!r}")


# ---------------------------------------------------------------------------
# Core Entities
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A network node."""

    id: str
    type: str
    capacity: Optional[float] = 1000.0
    elevation: Optional[float] = 50.0
    head: Optional[float] = None

    _VALID_TYPES = frozenset({"junction", "tank", "reservoir"})

    def __post_init__(self) -> None:
        if self.type not in self._VALID_TYPES:
            raise ValueError(
                f"Node type must be one of {sorted(self._VALID_TYPES)}, got {self.type!r}"
            )
        _validate_node_fields(self.type, self.capacity, self.elevation, self.head)


@dataclass
class Pipe:
    """A directed pipe connecting two nodes."""

    id: str
    start: str
    end: str
    has_pump: bool = False
    pump_power: float = 0.0
    length: Optional[float] = None
    diameter: Optional[float] = None
    roughness: Optional[float] = None
    valve_start: bool = False
    valve_end: bool = False

    def __post_init__(self) -> None:
        _validate_pipe_fields(
            self.length,
            self.diameter,
            self.roughness,
            self.pump_power,
            self.has_pump,
            self.valve_start,
            self.valve_end,
        )


# ---------------------------------------------------------------------------
# Network container
# ---------------------------------------------------------------------------


class Network:
    """Container for nodes and pipes with validation and ID management."""

    def __init__(self) -> None:
        self._nodes: List[Node] = []
        self._pipes: List[Pipe] = []
        # Per-prefix counters (track max issued number)
        self._counters: Dict[str, int] = {"J": 0, "T": 0, "R": 0, "P": 0}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes)

    @property
    def pipes(self) -> List[Pipe]:
        return list(self._pipes)

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_ids(self) -> set[str]:
        return {n.id for n in self._nodes}

    def _pipe_ids(self) -> set[str]:
        return {p.id for p in self._pipes}

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def _bump_counter(self, node_id: str) -> None:
        """Update counter so auto-generated IDs stay above *node_id*."""
        for prefix in ("J", "T", "R", "P"):
            if node_id.startswith(prefix):
                try:
                    num = int(node_id[len(prefix):])
                    if num > self._counters[prefix]:
                        self._counters[prefix] = num
                except ValueError:
                    pass
                break

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        type: str,  # noqa: A002
        id: Optional[str] = None,  # noqa: A002
        capacity: Optional[float] = 1000.0,
        elevation: Optional[float] = None,
        head: Optional[float] = None,
    ) -> Node:
        """Add a node and return it."""
        prefix_map = {"junction": "J", "tank": "T", "reservoir": "R"}
        prefix = prefix_map.get(type)
        if prefix is None:
            raise ValueError(
                f"Node type must be one of {sorted(prefix_map)}, got {type!r}"
            )

        _validate_node_fields(type, capacity, elevation, head)

        if id is None:
            id = self._next_id(prefix)
        elif id in self._node_ids():
            raise ValueError(f"Node ID {id!r} is already in use.")
        else:
            self._bump_counter(id)

        node = Node(id=id, type=type, capacity=capacity, elevation=elevation, head=head)
        self._nodes.append(node)
        return node

    def add_pipe(
        self,
        start: str,
        end: str,
        id: Optional[str] = None,  # noqa: A002
        has_pump: bool = False,
        pump_power: float = 0.0,
        length: Optional[float] = None,
        diameter: Optional[float] = None,
        roughness: Optional[float] = None,
        valve_start: bool = False,
        valve_end: bool = False,
    ) -> Pipe:
        """Add a pipe and return it."""
        if start == end:
            raise ValueError(f"Self-loop rejected: pipe from {start!r} to itself.")

        node_ids = self._node_ids()
        if start not in node_ids:
            raise ValueError(f"Unknown start node: {start!r}")
        if end not in node_ids:
            raise ValueError(f"Unknown end node: {end!r}")

        _validate_pipe_fields(
            length,
            diameter,
            roughness,
            pump_power,
            has_pump,
            valve_start,
            valve_end,
        )

        if id is None:
            id = self._next_id("P")
        elif id in self._pipe_ids():
            raise ValueError(f"Pipe ID {id!r} is already in use.")
        else:
            self._bump_counter(id)

        pipe = Pipe(
            id=id,
            start=start,
            end=end,
            has_pump=has_pump,
            pump_power=pump_power,
            length=length,
            diameter=diameter,
            roughness=roughness,
            valve_start=valve_start,
            valve_end=valve_end,
        )
        self._pipes.append(pipe)
        return pipe

    def update_node(self, node_id: str, **fields: Any) -> Node:
        """Update properties of an existing node.

        Allowed fields: 'elevation', 'capacity', 'head'.
        """
        node = next((n for n in self._nodes if n.id == node_id), None)
        if node is None:
            raise ValueError(f"Unknown node ID: {node_id!r}")

        allowed = {"elevation", "capacity", "head"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown field(s) for node: {sorted(unknown)}")

        new_capacity = fields["capacity"] if "capacity" in fields else node.capacity
        new_elevation = fields["elevation"] if "elevation" in fields else node.elevation
        new_head = fields["head"] if "head" in fields else node.head

        _validate_node_fields(node.type, new_capacity, new_elevation, new_head)

        if "capacity" in fields:
            node.capacity = new_capacity
        if "elevation" in fields:
            node.elevation = new_elevation
        if "head" in fields:
            node.head = new_head
        return node

    def update_pipe(self, pipe_id: str, **fields: Any) -> Pipe:
        """Update properties of an existing pipe.

        Allowed fields: 'length', 'diameter', 'roughness', 'has_pump', 'pump_power', 'valve_start', 'valve_end'.
        """
        pipe = next((p for p in self._pipes if p.id == pipe_id), None)
        if pipe is None:
            raise ValueError(f"Unknown pipe ID: {pipe_id!r}")

        allowed = {
            "length",
            "diameter",
            "roughness",
            "has_pump",
            "pump_power",
            "valve_start",
            "valve_end",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown field(s) for pipe: {sorted(unknown)}")

        new_length = fields["length"] if "length" in fields else pipe.length
        new_diameter = fields["diameter"] if "diameter" in fields else pipe.diameter
        new_roughness = fields["roughness"] if "roughness" in fields else pipe.roughness
        new_has_pump = fields["has_pump"] if "has_pump" in fields else pipe.has_pump
        new_pump_power = fields["pump_power"] if "pump_power" in fields else pipe.pump_power
        new_valve_start = fields["valve_start"] if "valve_start" in fields else pipe.valve_start
        new_valve_end = fields["valve_end"] if "valve_end" in fields else pipe.valve_end

        _validate_pipe_fields(
            new_length,
            new_diameter,
            new_roughness,
            new_pump_power,
            new_has_pump,
            new_valve_start,
            new_valve_end,
        )

        if "length" in fields:
            pipe.length = new_length
        if "diameter" in fields:
            pipe.diameter = new_diameter
        if "roughness" in fields:
            pipe.roughness = new_roughness
        if "has_pump" in fields:
            pipe.has_pump = new_has_pump
        if "pump_power" in fields:
            pipe.pump_power = new_pump_power
        if "valve_start" in fields:
            pipe.valve_start = new_valve_start
        if "valve_end" in fields:
            pipe.valve_end = new_valve_end
        return pipe

    def remove_pipe(self, pipe_id: str) -> None:
        """Remove a pipe by ID."""
        pipe = next((p for p in self._pipes if p.id == pipe_id), None)
        if pipe is None:
            raise ValueError(f"Unknown pipe ID: {pipe_id!r}")
        self._pipes.remove(pipe)

    def remove_node(self, node_id: str) -> None:
        """Remove a node by ID and any pipes attached to it."""
        node = next((n for n in self._nodes if n.id == node_id), None)
        if node is None:
            raise ValueError(f"Unknown node ID: {node_id!r}")
        self._nodes.remove(node)
        self._pipes = [p for p in self._pipes if p.start != node_id and p.end != node_id]

    def clear(self) -> None:
        """Remove all nodes and pipes and reset ID counters."""
        self._nodes.clear()
        self._pipes.clear()
        self._counters = {"J": 0, "T": 0, "R": 0, "P": 0}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain-dict representation (JSON-serialisable)."""
        pipes_data = []
        for p in self._pipes:
            p_dict = {
                "id": p.id,
                "start": p.start,
                "end": p.end,
                "has_pump": p.has_pump,
                "pump_power": p.pump_power,
                "length": p.length,
                "diameter": p.diameter,
                "roughness": p.roughness,
            }
            if p.valve_start:
                p_dict["valve_start"] = True
            if p.valve_end:
                p_dict["valve_end"] = True
            pipes_data.append(p_dict)

        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "capacity": n.capacity,
                    "elevation": n.elevation,
                    "head": n.head,
                }
                for n in self._nodes
            ],
            "pipes": pipes_data,
            "counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Network:
        """Re-create a :class:`Network` from a plain-dict representation."""
        net = cls()
        for nd in data.get("nodes", []):
            net.add_node(
                type=nd["type"],
                id=nd["id"],
                capacity=nd.get("capacity", 1000.0),
                elevation=nd.get("elevation", None),
                head=nd.get("head", None),
            )
        for pd_ in data.get("pipes", []):
            net.add_pipe(
                start=pd_["start"],
                end=pd_["end"],
                id=pd_["id"],
                has_pump=pd_.get("has_pump", False),
                pump_power=pd_.get("pump_power", 0.0),
                length=pd_.get("length", None),
                diameter=pd_.get("diameter", None),
                roughness=pd_.get("roughness", None),
                valve_start=bool(pd_.get("valve_start", False)),
                valve_end=bool(pd_.get("valve_end", False)),
            )
        if "counters" in data and isinstance(data["counters"], dict):
            net._counters.update(data["counters"])
        return net


    def __repr__(self) -> str:
        return f"Network({len(self._nodes)} nodes, {len(self._pipes)} pipes)"
