"""Widget for rendering directed water networks using Qt QGraphicsScene and QGraphicsView."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional

import networkx as nx

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QFontMetrics, QPainter,
    QPainterPath, QPolygonF, QImage, QPainterPathStroker
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsSimpleTextItem, QFileDialog
)

# ---------------------------------------------------------------------------
# Coordinate Conversion Constants & Helpers
# ---------------------------------------------------------------------------

SCALE: float = 300.0
NODE_RADIUS: float = 20.0


def model_to_scene(x: float, y: float, scale: float = SCALE) -> Tuple[float, float]:
    """Convert model/layout coordinate (y up) to Qt scene coordinate (y down)."""
    return (float(x) * scale, -float(y) * scale)


def scene_to_model(sx: float, sy: float, scale: float = SCALE) -> Tuple[float, float]:
    """Convert Qt scene coordinate (y down) to model/layout coordinate (y up)."""
    return (float(sx) / scale, -float(sy) / scale)


def measure_pipe_geometry(pipe_item: "PipeItem") -> Dict[str, object]:
    """Compute geometric measurements from the actual rendered QPainterPath of a PipeItem.

    Returns a dict with:
        'start_pt': QPointF, path.pointAtPercent(0.0)
        'end_pt': QPointF, path.pointAtPercent(1.0)
        'curve_midpoint': QPointF, path.pointAtPercent(0.5) in scene coordinates
        'chord_midpoint': QPointF, middle of start_pt and end_pt
        'chord_length': float, Euclidean distance between start_pt and end_pt
        'signed_displacement': float, perpendicular offset of curve_midpoint from chord
                              (positive = left of travel direction start -> end in screen view)
    """
    path = pipe_item.path
    start_pt = path.pointAtPercent(0.0)
    end_pt = path.pointAtPercent(1.0)
    curve_midpoint = path.pointAtPercent(0.5)

    chord_midpoint = QPointF((start_pt.x() + end_pt.x()) / 2.0, (start_pt.y() + end_pt.y()) / 2.0)
    chord_dx = end_pt.x() - start_pt.x()
    chord_dy = end_pt.y() - start_pt.y()
    chord_length = math.hypot(chord_dx, chord_dy)

    if chord_length < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = chord_dx / chord_length, chord_dy / chord_length

    # Left normal in screen view (positive = left of travel direction start -> end)
    nx_left, ny_left = uy, -ux

    vx = curve_midpoint.x() - chord_midpoint.x()
    vy = curve_midpoint.y() - chord_midpoint.y()

    signed_displacement = vx * nx_left + vy * ny_left

    return {
        "start_pt": start_pt,
        "end_pt": end_pt,
        "curve_midpoint": curve_midpoint,
        "chord_midpoint": chord_midpoint,
        "chord_length": chord_length,
        "signed_displacement": signed_displacement,
    }


# ---------------------------------------------------------------------------
# Graphics Items
# ---------------------------------------------------------------------------


class NodeItem(QGraphicsItem):
    """Movable circular node item with centered ID label, hover cursor, and live drag."""

    def __init__(self, node_id: str, node_type: str, pos: QPointF, graph_view: Optional["GraphView"] = None, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.node_type = node_type.lower()
        self._graph_view = graph_view
        self.setPos(pos)
        self.setZValue(10)

        self._highlighted = False
        self._selected = False
        self._press_scene_pos: Optional[QPointF] = None
        self._press_item_pos: Optional[QPointF] = None
        self._is_dragging = False

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)

        self.color_map = {
            "junction": QColor("#87CEEB"),
            "tank": QColor("#FFD700"),
            "reservoir": QColor("#FA8072"),
        }
        self.base_color = self.color_map.get(self.node_type, QColor("#CCCCCC"))

    def set_highlighted(self, val: bool):
        if self._highlighted != val:
            self._highlighted = val
            self.update()

    def set_selected(self, val: bool):
        if self._selected != val:
            self._selected = val
            self.update()

    def setSelected(self, selected: bool):
        super().setSelected(selected)
        self.set_selected(selected)

    def boundingRect(self) -> QRectF:
        r = NODE_RADIUS + 8.0
        return QRectF(-r, -r, 2.0 * r, 2.0 * r)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            if getattr(self, "_graph_view", None) is not None:
                self._graph_view._on_node_moved(self.node_id, self.pos())
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self._press_scene_pos = event.scenePos()
        self._press_item_pos = self.pos()
        self._is_dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._press_scene_pos is not None:
            if (event.scenePos() - self._press_scene_pos).manhattanLength() > 2.0:
                self._is_dragging = True

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        was_dragging = self._is_dragging
        press_item_pos = self._press_item_pos
        self._is_dragging = False
        self._press_scene_pos = None
        self._press_item_pos = None

        if was_dragging and press_item_pos is not None and (self.pos() - press_item_pos).manhattanLength() > 0.5:
            if self._graph_view:
                self._graph_view._on_node_drag_finished(self.node_id)
        elif not was_dragging:
            if self._graph_view:
                self._graph_view._on_node_clicked(self.node_id)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        # Base node circle fill
        painter.setBrush(QBrush(self.base_color))

        if self._selected and self._highlighted:
            # Dual rings: inner red highlight and outer blue selection ring
            painter.setPen(QPen(QColor("red"), 3.0))
            painter.drawEllipse(QPointF(0, 0), NODE_RADIUS, NODE_RADIUS)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#0078D7"), 3.5))
            painter.drawEllipse(QPointF(0, 0), NODE_RADIUS + 4.0, NODE_RADIUS + 4.0)
        elif self._selected:
            # Blue selection ring
            painter.setPen(QPen(QColor("#0078D7"), 3.5))
            painter.drawEllipse(QPointF(0, 0), NODE_RADIUS, NODE_RADIUS)
        elif self._highlighted:
            # Red outline
            painter.setPen(QPen(QColor("red"), 3.0))
            painter.drawEllipse(QPointF(0, 0), NODE_RADIUS, NODE_RADIUS)
        else:
            # Normal black outline
            painter.setPen(QPen(QColor("#000000"), 1.5))
            painter.drawEllipse(QPointF(0, 0), NODE_RADIUS, NODE_RADIUS)

        # Center node ID label
        painter.setPen(QColor("#000000"))
        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(-NODE_RADIUS, -NODE_RADIUS, 2 * NODE_RADIUS, 2 * NODE_RADIUS),
            Qt.AlignCenter,
            self.node_id
        )


class BendHandleItem(QGraphicsItem):
    """Small round cosmetic handle at the curve midpoint for bending pipes."""

    def __init__(self, pipe_item: "PipeItem", parent=None):
        super().__init__(parent or pipe_item)
        self.pipe_item = pipe_item
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(25)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self.radius = 4.0  # 8 px cosmetic diameter
        self.setVisible(False)
        self._dragging = False
        self._initial_bend: float = 0.0

    def boundingRect(self) -> QRectF:
        r = self.radius + 3.0
        return QRectF(-r, -r, 2.0 * r, 2.0 * r)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#0078D7")))
        painter.setPen(QPen(QColor("white"), 1.5))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

    def hoverEnterEvent(self, event):
        self.pipe_item._is_hovered = True
        self.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        if not self.pipe_item._selected and not self.pipe_item.isUnderMouse():
            self.pipe_item._is_hovered = False
            self.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
            self._dragging = True
            self._initial_bend = self.pipe_item.bend_ratio
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            event.accept()
            self.pipe_item.handle_drag_to(event.scenePos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            event.accept()
            self._dragging = False
            self.pipe_item.handle_drag_finished(self._initial_bend)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
            self.pipe_item.reset_bend()
        else:
            super().mouseDoubleClickEvent(event)


class ValveSymbolItem(QGraphicsItem):
    """Small bow-tie valve symbol near a pipe end."""

    def __init__(self, pipe_item: "PipeItem", end_name: str, parent=None):
        super().__init__(parent or pipe_item)
        self.pipe_item = pipe_item
        self.end_name = end_name
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(15)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"Valve at {end_name} of {pipe_item.pipe_id}")
        self.angle_deg = 0.0

    def boundingRect(self) -> QRectF:
        return QRectF(-10, -10, 20, 20)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.save()
        painter.rotate(self.angle_deg)
        painter.setPen(QPen(QColor("#222222"), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        left_poly = QPolygonF([QPointF(-6, -4.5), QPointF(0, 0), QPointF(-6, 4.5)])
        right_poly = QPolygonF([QPointF(6, -4.5), QPointF(0, 0), QPointF(6, 4.5)])
        painter.drawPolygon(left_poly)
        painter.drawPolygon(right_poly)
        painter.restore()

    def mousePressEvent(self, event):
        self.pipe_item.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.pipe_item.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.pipe_item.mouseReleaseEvent(event)


class PipeItem(QGraphicsItem):
    """Directed pipe item stopping at node boundaries with arrow, curve, and live bend handle."""

    def __init__(
        self,
        pipe_id: str,
        start_node_id: str,
        end_node_id: str,
        start_pos: QPointF,
        end_pos: QPointF,
        has_pump: bool = False,
        label_text: str = "",
        curvature: float = 0.0,
        bend_ratio: float = 0.0,
        graph_view: Optional["GraphView"] = None,
        valve_start: bool = False,
        valve_end: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.pipe_id = pipe_id
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.has_pump = has_pump
        self.label_text = label_text or pipe_id
        self.auto_curvature = curvature
        self.bend_ratio = bend_ratio
        self.curvature = self.auto_curvature + self.bend_ratio
        self._graph_view = graph_view
        self.setZValue(5)

        self._highlighted = False
        self._selected = False
        self._is_hovered = False
        self.setAcceptHoverEvents(True)

        self._label_box = QRectF()
        self._label_dragging = False
        self._pipe_pressed = False
        self._pipe_press_scene = QPointF()
        self._initial_drag_bend = 0.0

        self.valve_start = valve_start
        self.valve_end = valve_end
        self.valve_start_item = ValveSymbolItem(self, "start") if valve_start else None
        self.valve_end_item = ValveSymbolItem(self, "end") if valve_end else None

        self.handle_item = BendHandleItem(self)
        self._build_geometry()

    def set_start_pos(self, pos: QPointF):
        self.prepareGeometryChange()
        self.start_pos = pos
        self._build_geometry()
        self.update()

    def set_end_pos(self, pos: QPointF):
        self.prepareGeometryChange()
        self.end_pos = pos
        self._build_geometry()
        self.update()

    def _build_geometry(self):
        p1 = self.start_pos
        p2 = self.end_pos
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        nodes_len = math.hypot(dx, dy)
        if nodes_len < 1e-4:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / nodes_len, dy / nodes_len
        # Left normal in screen view (positive curvature is left of travel direction)
        nx_left, ny_left = uy, -ux

        # End points clipped at node boundaries along the direct line between nodes
        if nodes_len > 2.0 * NODE_RADIUS:
            start_pt = QPointF(p1.x() + ux * NODE_RADIUS, p1.y() + uy * NODE_RADIUS)
            end_pt = QPointF(p2.x() - ux * NODE_RADIUS, p2.y() - uy * NODE_RADIUS)
        else:
            start_pt = p1
            end_pt = p2

        chord_dx = end_pt.x() - start_pt.x()
        chord_dy = end_pt.y() - start_pt.y()
        chord_len = math.hypot(chord_dx, chord_dy)
        if chord_len < 1e-4:
            chord_len = 1.0

        # Chord midpoint
        mx = (start_pt.x() + end_pt.x()) / 2.0
        my = (start_pt.y() + end_pt.y()) / 2.0

        self.curvature = self.auto_curvature + self.bend_ratio
        if abs(self.curvature) < 1e-4:
            qx, qy = mx, my
        else:
            # For quadratic Bezier, curve midpoint B(0.5) = M + 0.5 * (Q - M).
            # To have curve midpoint displaced by curvature * chord_len * n_left,
            # Q - M = 2.0 * curvature * chord_len * n_left.
            qx = mx + 2.0 * self.curvature * chord_len * nx_left
            qy = my + 2.0 * self.curvature * chord_len * ny_left

        self.path = QPainterPath()
        self.path.moveTo(start_pt)
        ctrl_pt = QPointF(qx, qy)
        if abs(self.curvature) < 1e-4:
            self.path.lineTo(end_pt)
        else:
            self.path.quadTo(ctrl_pt, end_pt)

        # Arrowhead at end_pt pointing along tangent from ctrl_pt
        tx = end_pt.x() - ctrl_pt.x()
        ty = end_pt.y() - ctrl_pt.y()
        tlen = math.hypot(tx, ty)
        if tlen > 1e-4:
            tux, tuy = tx / tlen, ty / tlen
        else:
            tux, tuy = ux, uy
        tnx, tny = -tuy, tux

        arrow_len = 13.0
        arrow_half_width = 5.0
        tip = end_pt
        left_barb = QPointF(
            tip.x() - tux * arrow_len + tnx * arrow_half_width,
            tip.y() - tuy * arrow_len + tny * arrow_half_width,
        )
        right_barb = QPointF(
            tip.x() - tux * arrow_len - tnx * arrow_half_width,
            tip.y() - tuy * arrow_len - tny * arrow_half_width,
        )
        self.arrow_poly = QPolygonF([tip, left_barb, right_barb])

        # Exact midpoint on the quadratic Bezier curve
        self.curve_midpoint = self.path.pointAtPercent(0.5)
        self.label_pos = self.curve_midpoint

        if hasattr(self, "handle_item") and self.handle_item:
            self.handle_item.setPos(self.curve_midpoint)

        # Precompute label bounding box
        font = QFont("Segoe UI", 8)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(self.label_text)
        th = fm.height()
        self._label_box = QRectF(
            self.label_pos.x() - tw / 2.0 - 4,
            self.label_pos.y() - th / 2.0 - 2,
            tw + 8,
            th + 4,
        )

        if hasattr(self, "valve_start_item") and self.valve_start_item:
            v_pt = self.path.pointAtPercent(0.08)
            self.valve_start_item.setPos(v_pt)
            p_prev = self.path.pointAtPercent(0.06)
            p_next = self.path.pointAtPercent(0.10)
            self.valve_start_item.angle_deg = math.degrees(math.atan2(p_next.y() - p_prev.y(), p_next.x() - p_prev.x()))

        if hasattr(self, "valve_end_item") and self.valve_end_item:
            v_pt = self.path.pointAtPercent(0.88)
            self.valve_end_item.setPos(v_pt)
            p_prev = self.path.pointAtPercent(0.86)
            p_next = self.path.pointAtPercent(0.90)
            self.valve_end_item.angle_deg = math.degrees(math.atan2(p_next.y() - p_prev.y(), p_next.x() - p_prev.x()))

    def set_valves(self, valve_start: bool, valve_end: bool):
        self.valve_start = valve_start
        self.valve_end = valve_end
        if valve_start and self.valve_start_item is None:
            self.valve_start_item = ValveSymbolItem(self, "start")
        elif not valve_start and self.valve_start_item is not None:
            if self.scene():
                self.scene().removeItem(self.valve_start_item)
            self.valve_start_item = None

        if valve_end and self.valve_end_item is None:
            self.valve_end_item = ValveSymbolItem(self, "end")
        elif not valve_end and self.valve_end_item is not None:
            if self.scene():
                self.scene().removeItem(self.valve_end_item)
            self.valve_end_item = None
        self._build_geometry()
        self.update()

    def hoverEnterEvent(self, event):
        self._is_hovered = True
        self.handle_item.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        self._is_hovered = False
        if not self._selected and not self.handle_item.isUnderMouse():
            self.handle_item.setVisible(False)

    def set_highlighted(self, val: bool):
        if self._highlighted != val:
            self._highlighted = val
            self.update()

    def set_selected(self, val: bool):
        if self._selected != val:
            self._selected = val
            super().setSelected(val)
            self.handle_item.setVisible(self._selected or self._is_hovered)
            self.update()

    def setSelected(self, val: bool):
        self.set_selected(val)

    def isSelected(self) -> bool:
        return self._selected

    def handle_drag_to(self, scene_pos: QPointF):
        p1 = self.start_pos
        p2 = self.end_pos
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        nodes_len = math.hypot(dx, dy)
        if nodes_len < 1e-4:
            return

        ux, uy = dx / nodes_len, dy / nodes_len
        nx_left, ny_left = uy, -ux

        if nodes_len > 2.0 * NODE_RADIUS:
            start_pt = QPointF(p1.x() + ux * NODE_RADIUS, p1.y() + uy * NODE_RADIUS)
            end_pt = QPointF(p2.x() - ux * NODE_RADIUS, p2.y() - uy * NODE_RADIUS)
        else:
            start_pt = p1
            end_pt = p2

        chord_dx = end_pt.x() - start_pt.x()
        chord_dy = end_pt.y() - start_pt.y()
        chord_len = math.hypot(chord_dx, chord_dy)
        if chord_len < 1e-4:
            return

        mx = (start_pt.x() + end_pt.x()) / 2.0
        my = (start_pt.y() + end_pt.y()) / 2.0

        vx = scene_pos.x() - mx
        vy = scene_pos.y() - my

        perp_offset = vx * nx_left + vy * ny_left
        total_curvature = perp_offset / chord_len

        raw_bend = total_curvature - self.auto_curvature
        clamped_bend = max(-1.5, min(1.5, raw_bend))
        if abs(clamped_bend) < 1e-4:
            clamped_bend = 0.0

        self.bend_ratio = clamped_bend
        self.curvature = self.auto_curvature + self.bend_ratio

        self.prepareGeometryChange()
        self._build_geometry()
        self.update()

        if self._graph_view:
            self._graph_view._on_pipe_bend_updated(self.pipe_id, self.bend_ratio)

    def handle_drag_finished(self, initial_bend: float):
        if abs(self.bend_ratio - initial_bend) > 1e-4:
            if self._graph_view:
                self._graph_view._on_pipe_bend_finished(self.pipe_id)

    def reset_bend(self):
        old_bend = self.bend_ratio
        self.bend_ratio = 0.0
        self.curvature = self.auto_curvature
        self.prepareGeometryChange()
        self._build_geometry()
        self.update()
        if self._graph_view:
            self._graph_view._on_pipe_bend_updated(self.pipe_id, 0.0)
            if abs(old_bend) > 1e-4:
                self._graph_view._on_pipe_bend_finished(self.pipe_id)

    def set_bend_ratio(self, val: float):
        self.bend_ratio = max(-1.5, min(1.5, float(val)))
        if abs(self.bend_ratio) < 1e-4:
            self.bend_ratio = 0.0
        self.curvature = self.auto_curvature + self.bend_ratio
        self.prepareGeometryChange()
        self._build_geometry()
        self.update()

    def boundingRect(self) -> QRectF:
        rect = self.path.boundingRect()
        rect = rect.united(self.arrow_poly.boundingRect())
        rect = rect.united(self._label_box)
        return rect.adjusted(-30, -30, 30, 30)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        p = stroker.createStroke(self.path)
        p.addPolygon(self.arrow_poly)
        p.addRect(self._label_box)
        return p

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self, "_label_box") and self._label_box.contains(event.pos()):
                event.accept()
                self._label_dragging = True
                self._initial_drag_bend = self.bend_ratio
                return
            else:
                event.accept()
                self._pipe_pressed = True
                self._pipe_press_scene = event.scenePos()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._label_dragging:
            event.accept()
            self.handle_drag_to(event.scenePos())
            return
        elif self._pipe_pressed:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._label_dragging:
                event.accept()
                self._label_dragging = False
                self.handle_drag_finished(self._initial_drag_bend)
                return
            elif self._pipe_pressed:
                event.accept()
                self._pipe_pressed = False
                if (event.scenePos() - self._pipe_press_scene).manhattanLength() < 5.0:
                    if self._graph_view:
                        self._graph_view._on_pipe_clicked(self.pipe_id)
                return
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Curve
        if self._selected and self._highlighted:
            # Dual: thicker blue line beneath red line
            painter.setPen(QPen(QColor("#0078D7"), 6.0, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self.path)

            line_pen = QPen(QColor("red"), 3.0, Qt.SolidLine)
            color = QColor("red")
        elif self._selected:
            color = QColor("#0078D7")
            line_pen = QPen(color, 4.0, Qt.SolidLine)
        elif self._highlighted:
            color = QColor("red")
            line_pen = QPen(color, 3.5, Qt.SolidLine)
        else:
            color = QColor("#DC143C") if self.has_pump else QColor("#333333")
            style = Qt.DashLine if self.has_pump else Qt.SolidLine
            line_pen = QPen(color, 2.0, style)

        line_pen.setCosmetic(True)
        if line_pen.widthF() < 2.0:
            line_pen.setWidthF(2.0)

        painter.setPen(line_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path)

        # 2. Arrowhead
        arrow_color = QColor("#0078D7") if (self._selected and not self._highlighted) else (QColor("red") if self._highlighted else color)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(arrow_color))
        painter.drawPolygon(self.arrow_poly)

        # 3. Label with white rounded background
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawRoundedRect(self._label_box, 3, 3)

        painter.setPen(color)
        painter.drawText(self._label_box, Qt.AlignCenter, self.label_text)


class LegendItem(QGraphicsItem):
    """Overlay legend drawn in top-right of view, ignoring transformations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(100)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, 120, 155)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#CCCCCC"), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawRoundedRect(0, 0, 120, 155, 4, 4)

        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        entries = [
            ("Junction", QColor("#87CEEB"), "node"),
            ("Tank", QColor("#FFD700"), "node"),
            ("Reservoir", QColor("#FA8072"), "node"),
            ("Pipe", QColor("#333333"), "pipe"),
            ("Pump", QColor("#DC143C"), "pump"),
            ("Valve", QColor("#333333"), "valve"),
        ]

        y = 10
        for label, color, kind in entries:
            if kind == "node":
                painter.setPen(QPen(QColor("#000000"), 1))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(12, y + 2, 10, 10)
            elif kind == "pipe":
                painter.setPen(QPen(color, 2, Qt.SolidLine))
                painter.drawLine(8, y + 7, 26, y + 7)
            elif kind == "pump":
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawLine(8, y + 7, 26, y + 7)
            elif kind == "valve":
                painter.setPen(QPen(QColor("#222222"), 1.2))
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                lp = QPolygonF([QPointF(11, y + 4), QPointF(17, y + 7), QPointF(11, y + 10)])
                rp = QPolygonF([QPointF(23, y + 4), QPointF(17, y + 7), QPointF(23, y + 10)])
                painter.drawPolygon(lp)
                painter.drawPolygon(rp)

            painter.setPen(QColor("#000000"))
            painter.drawText(34, y + 12, label)
            y += 24


# ---------------------------------------------------------------------------
# Custom View Widget
# ---------------------------------------------------------------------------


class NetworkGraphicsView(QGraphicsView):
    """QGraphicsView with smooth wheel zoom around cursor and hand drag panning."""

    def __init__(self, scene: QGraphicsScene, graph_view: "GraphView", parent=None):
        super().__init__(scene, parent)
        self._graph_view = graph_view
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewport().setCursor(Qt.OpenHandCursor)
        self._mouse_press_pos = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._graph_view._user_manually_navigated = True
        self._graph_view.update_legend_position()

    def scrollContentsBy(self, dx: int, dy: int):
        super().scrollContentsBy(dx, dy)
        try:
            self._graph_view.update_legend_position()
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        self._graph_view._apply_fit_if_pending()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._graph_view._apply_fit_if_pending()
        try:
            self._graph_view.update_legend_position()
        except Exception:
            pass

    def mousePressEvent(self, event):
        self._mouse_press_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mouse_press_pos is not None:
            cur_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            if (cur_pos - self._mouse_press_pos).manhattanLength() > 4:
                self._graph_view._user_manually_navigated = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._mouse_press_pos is not None:
            cur_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            diff = (cur_pos - self._mouse_press_pos).manhattanLength()
            self._mouse_press_pos = None
            if diff < 4:
                item = self.itemAt(cur_pos)
                if item is None or item is self._graph_view.legend_item or isinstance(item, QGraphicsSimpleTextItem):
                    self._graph_view._on_background_clicked()


# ---------------------------------------------------------------------------
# Main GraphView Widget
# ---------------------------------------------------------------------------


class GraphView(QWidget):
    """Widget displaying the directed water network graph using QGraphicsScene/View."""

    positions_changed = Signal()
    edge_bends_changed = Signal()
    node_clicked = Signal(str)
    pipe_clicked = Signal(str)
    background_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pos: Dict[str, Tuple[float, float]] = {}
        self._edge_bends: Dict[str, float] = {}
        self._highlight_ids: Set[str] = set()
        self._selected_node_id: Optional[str] = None
        self._selected_pipe_id: Optional[str] = None
        self._node_pipes: Dict[str, List[PipeItem]] = defaultdict(list)
        self._current_network = None
        self._last_drawn_node_count = 0
        self._pending_fit = True
        self._user_manually_navigated = False

        self._init_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_fit_if_pending()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._apply_fit_if_pending)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_fit_if_pending()

    def _is_viewport_valid(self) -> bool:
        vp = self.view.viewport()
        return self.isVisible() and vp.width() > 50 and vp.height() > 50

    def _apply_fit_if_pending(self):
        if self._pending_fit and self._is_viewport_valid():
            self.fit_in_view()
            self._pending_fit = False

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top Control Bar
        top_bar = QHBoxLayout()
        self.relayout_btn = QPushButton("Re-layout")
        self.relayout_btn.setToolTip("Recalculate layout for all nodes")
        self.relayout_btn.clicked.connect(self._on_relayout)
        top_bar.addWidget(self.relayout_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Fit all network elements in view")
        self.fit_btn.clicked.connect(self.fit_in_view)
        top_bar.addWidget(self.fit_btn)

        self.save_img_btn = QPushButton("Save image...")
        self.save_img_btn.setToolTip("Export current network drawing to PNG image")
        self.save_img_btn.clicked.connect(self._on_save_image)
        top_bar.addWidget(self.save_img_btn)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Graphics Scene and View
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor("#FFFFFF")))
        self.view = NetworkGraphicsView(self.scene, self, self)
        self.view.setBackgroundBrush(QBrush(QColor("#FFFFFF")))
        layout.addWidget(self.view)

        self.legend_item = LegendItem()
        self.scene.addItem(self.legend_item)
        self.legend_item.setVisible(False)

        self.hint_item: Optional[QGraphicsSimpleTextItem] = None
        self._show_empty_hint()

    def _show_empty_hint(self):
        self.scene.clear()
        self._node_pipes.clear()
        self.hint_item = QGraphicsSimpleTextItem("Add nodes and pipes to view network graph")
        self.hint_item.setFont(QFont("Segoe UI", 12))
        self.hint_item.setBrush(QBrush(QColor("#888888")))
        br = self.hint_item.boundingRect()
        self.hint_item.setPos(-br.width() / 2.0, -br.height() / 2.0)
        self.scene.addItem(self.hint_item)

        self.legend_item = LegendItem()
        self.scene.addItem(self.legend_item)
        self.legend_item.setVisible(False)

    def _on_relayout(self):
        self._pos.clear()
        self._edge_bends.clear()
        self._pending_fit = True
        if self._current_network:
            self.draw_network(self._current_network)

    def _on_save_image(self):
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Network Image", "network.png", "PNG Images (*.png)"
        )
        if path_str:
            self.export_image(path_str)

    def get_positions(self) -> Dict[str, Tuple[float, float]]:
        """Return current canvas node positions in model space [-1, 1]."""
        return dict(self._pos)

    def set_positions(self, pos: Dict[str, Tuple[float, float]]) -> None:
        """Set canvas node positions in model space and refresh drawing."""
        self._pos = dict(pos) if pos else {}
        self._pending_fit = True
        if pos and self._current_network:
            self.draw_network(self._current_network)

    def get_edge_bends(self) -> Dict[str, float]:
        """Return mapping of pipe IDs to dimensionless bend ratios, omitting zeros."""
        return {pid: round(b, 6) for pid, b in self._edge_bends.items() if abs(b) > 1e-6}

    def set_edge_bends(self, bends: Optional[Dict[str, float]]) -> None:
        """Set edge bends silently without emitting signals."""
        self._edge_bends = {
            str(pid): float(val) for pid, val in bends.items() if abs(float(val)) > 1e-6
        } if bends else {}
        for p in self.pipe_items():
            p.set_bend_ratio(self._edge_bends.get(p.pipe_id, 0.0))

    def highlight(self, ids: List[str]) -> None:
        """Highlight specified node/pipe IDs with a red outline / thick red line."""
        self._highlight_ids = set(ids)
        for item in self.scene.items():
            if isinstance(item, NodeItem):
                item.set_highlighted(item.node_id in self._highlight_ids)
            elif isinstance(item, PipeItem):
                item.set_highlighted(item.pipe_id in self._highlight_ids)

    def clear_highlight(self) -> None:
        """Clear all active graph highlights."""
        if self._highlight_ids:
            self._highlight_ids.clear()
            for item in self.scene.items():
                if isinstance(item, (NodeItem, PipeItem)):
                    item.set_highlighted(False)

    def set_selection(self, node_id: Optional[str] = None, pipe_id: Optional[str] = None):
        """Set visual blue selection for a node or pipe, clearing previous selection."""
        self._selected_node_id = node_id
        self._selected_pipe_id = pipe_id
        for it in self.scene.items():
            if isinstance(it, NodeItem):
                it.set_selected(it.node_id == node_id)
            elif isinstance(it, PipeItem):
                it.set_selected(it.pipe_id == pipe_id)

    def clear_selection(self):
        """Clear visual selection on all graph items."""
        self._selected_node_id = None
        self._selected_pipe_id = None
        for it in self.scene.items():
            if isinstance(it, (NodeItem, PipeItem)):
                it.set_selected(False)

    def node_items(self) -> List[NodeItem]:
        """Return all NodeItem instances in the scene."""
        return [it for it in self.scene.items() if isinstance(it, NodeItem)]

    def pipe_items(self) -> List[PipeItem]:
        """Return all PipeItem instances in the scene."""
        return [it for it in self.scene.items() if isinstance(it, PipeItem)]

    def _on_node_moved(self, node_id: str, new_scene_pos: QPointF):
        """Called live while a node is being dragged."""
        mx, my = scene_to_model(new_scene_pos.x(), new_scene_pos.y())
        self._pos[node_id] = (mx, my)
        for p_item in self._node_pipes.get(node_id, []):
            if p_item.start_node_id == node_id:
                p_item.set_start_pos(new_scene_pos)
            if p_item.end_node_id == node_id:
                p_item.set_end_pos(new_scene_pos)

    def _on_node_drag_finished(self, node_id: str):
        """Called once on mouse release after a real node drag."""
        self.positions_changed.emit()

    def _on_pipe_bend_updated(self, pipe_id: str, bend: float):
        """Called live while a pipe is being bent."""
        if abs(bend) > 1e-6:
            self._edge_bends[pipe_id] = bend
        else:
            self._edge_bends.pop(pipe_id, None)

    def _on_pipe_bend_finished(self, pipe_id: str):
        """Called once on mouse release after a real pipe bend change."""
        self.edge_bends_changed.emit()

    def _on_node_clicked(self, node_id: str):
        """Called on a plain click on a node."""
        self.node_clicked.emit(node_id)

    def _on_pipe_clicked(self, pipe_id: str):
        """Called on a plain click on a pipe."""
        self.pipe_clicked.emit(pipe_id)

    def _on_background_clicked(self):
        """Called on a plain click on the empty canvas background."""
        self.background_clicked.emit()

    def update_legend_position(self):
        """Keep legend pinned to the top-right corner of the view without zooming."""
        try:
            if not self._current_network or not self._current_network.nodes:
                if hasattr(self, "legend_item") and self.legend_item:
                    self.legend_item.setVisible(False)
                return
            if hasattr(self, "legend_item") and self.legend_item:
                self.legend_item.setVisible(True)
                vw = self.view.viewport().width()
                if vw <= 50:
                    return
                lw = self.legend_item.boundingRect().width()
                target_scene_pt = self.view.mapToScene(max(10, int(round(vw - lw - 8.0))), 8)
                self.legend_item.setPos(target_scene_pt)
        except (RuntimeError, AttributeError):
            pass

    def fit_in_view(self, pad_px: float = 30.0):
        """Fit all graph elements nicely in the viewport reserving legend rectangle."""
        self._user_manually_navigated = False
        nodes = [it for it in self.scene.items() if isinstance(it, NodeItem)]
        pipes = [it for it in self.scene.items() if isinstance(it, PipeItem)]
        if not nodes:
            return

        content_rect = nodes[0].sceneBoundingRect()
        for it in nodes[1:]:
            content_rect = content_rect.united(it.sceneBoundingRect())

        for p in pipes:
            p_box = p.path.boundingRect().united(p.arrow_poly.boundingRect()).united(p._label_box)
            content_rect = content_rect.united(p_box)

        if content_rect.isEmpty() or content_rect.width() <= 0 or content_rect.height() <= 0:
            return

        vp = self.view.viewport()
        vw, vh = vp.width(), vp.height()
        if vw <= 50 or vh <= 50:
            return

        legend_visible = hasattr(self, "legend_item") and self.legend_item and self.legend_item.isVisible()
        # Reserve legend rectangle: right margin = legend width + 16 px (136 px)
        lw = self.legend_item.boundingRect().width() if hasattr(self, "legend_item") and self.legend_item else 120.0
        right_pad = (lw + 16.0) if legend_visible else pad_px
        left_pad = pad_px
        top_pad = pad_px
        bottom_pad = pad_px

        nc = content_rect.center()
        max_w_left = max(nc.x() - content_rect.left(), 1.0)
        max_w_right = max(content_rect.right() - nc.x(), 1.0)
        max_h_top = max(nc.y() - content_rect.top(), 1.0)
        max_h_bottom = max(content_rect.bottom() - nc.y(), 1.0)

        scale_l = max(10.0, float(vw) / 2.0 - left_pad) / max_w_left
        scale_r = max(10.0, float(vw) / 2.0 - right_pad) / max_w_right
        scale_t = max(10.0, float(vh) / 2.0 - top_pad) / max_h_top
        scale_b = max(10.0, float(vh) / 2.0 - bottom_pad) / max_h_bottom

        scale = min(scale_l, scale_r, scale_t, scale_b)

        pad_scene = max(vw, vh) / scale
        self.scene.setSceneRect(content_rect.adjusted(-pad_scene, -pad_scene, pad_scene, pad_scene))

        self.view.resetTransform()
        self.view.scale(scale, scale)
        self.view.centerOn(nc)
        self.update_legend_position()

    def render_to_image(self, scale: float = 2.0, margin: float = 24.0) -> QImage:
        """Render the network graph cropped to content plus margin on a white background."""
        nodes = [it for it in self.scene.items() if isinstance(it, NodeItem)]
        pipes = [it for it in self.scene.items() if isinstance(it, PipeItem)]
        graph_items = nodes + pipes

        if not graph_items:
            rect = self.hint_item.sceneBoundingRect() if self.hint_item else QRectF(-150, -50, 300, 100)
            export_rect = rect.adjusted(-margin, -margin, margin, margin)
            w = max(100, int(round(export_rect.width() * scale)))
            h = max(60, int(round(export_rect.height() * scale)))
            image = QImage(w, h, QImage.Format_ARGB32)
            image.fill(QColor("white"))
            painter = QPainter(image)
            painter.setRenderHints(
                QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
            )
            self.scene.render(painter, target=QRectF(0, 0, w, h), source=export_rect)
            painter.end()
            return image

        graph_rect = graph_items[0].sceneBoundingRect()
        for it in graph_items[1:]:
            graph_rect = graph_rect.united(it.sceneBoundingRect())

        # Determine legend placement inside the exported image without covering nodes, pipes, or labels
        lw = self.legend_item.boundingRect().width()
        lh = self.legend_item.boundingRect().height()

        lx = graph_rect.right() + 16.0
        ly = graph_rect.top()
        legend_rect = QRectF(lx, ly, lw, lh)

        self._last_export_legend_rect = legend_rect
        self.legend_item.setVisible(True)
        self.legend_item.setPos(lx, ly)

        total_content_rect = graph_rect.united(legend_rect)
        export_rect = total_content_rect.adjusted(-margin, -margin, margin, margin)

        w = int(round(export_rect.width() * scale))
        h = int(round(export_rect.height() * scale))

        image = QImage(w, h, QImage.Format_ARGB32)
        image.fill(QColor("white"))

        painter = QPainter(image)
        painter.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform
        )
        self.scene.render(painter, target=QRectF(0, 0, w, h), source=export_rect)
        painter.end()

        self.update_legend_position()
        return image

    def export_image(self, file_path: str | Path, *args, scale: float = 2.0, margin: float = 24.0, **kwargs) -> bool:
        """Export the cropped network graph to a PNG image."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.render_to_image(scale=scale, margin=margin)
        return image.save(str(file_path), "PNG")

    save_image = export_image

    def draw_network(self, network):
        """Draw the given Network instance on the QGraphicsScene."""
        is_first_draw = (
            self._current_network is None
            or self._current_network is not network
            or self._last_drawn_node_count == 0
        )
        if is_first_draw:
            self._pending_fit = True
            self._user_manually_navigated = False

        self._current_network = network
        self._last_drawn_node_count = len(network.nodes)

        if not network.nodes:
            self._show_empty_hint()
            self._pending_fit = True
            return

        self.scene.clear()
        self._node_pipes.clear()

        # Build NetworkX MultiDiGraph for layout
        G = nx.MultiDiGraph()
        for n in network.nodes:
            G.add_node(n.id, type=n.type)
        for p in network.pipes:
            label = f"{p.id} ({p.pump_power:.0f}kW)" if p.has_pump else p.id
            G.add_edge(p.start, p.end, id=p.id, has_pump=p.has_pump, power=p.pump_power, label=label)

        # Update node layout preserving existing positions
        current_nodes = set(G.nodes())
        for nid in list(self._pos.keys()):
            if nid not in current_nodes:
                del self._pos[nid]

        if not self._pos:
            if hasattr(network, "default_positions") and network.default_positions:
                self._pos = dict(network.default_positions)
            else:
                self._pos = nx.spring_layout(G, seed=42)
        else:
            fixed_nodes = [n for n in G.nodes() if n in self._pos]
            if len(fixed_nodes) < len(G.nodes()):
                import numpy as np
                rng = np.random.default_rng(42)
                pos_init = dict(self._pos)
                for n in G.nodes():
                    if n not in pos_init:
                        pos_init[n] = rng.uniform(-1, 1, size=2)
                new_pos = nx.spring_layout(G, pos=pos_init, fixed=fixed_nodes, seed=42)
                self._pos.update(new_pos)

        # Map to scene coordinates
        scene_node_positions: Dict[str, QPointF] = {}
        for nid, (mx, my) in self._pos.items():
            sx, sy = model_to_scene(mx, my)
            scene_node_positions[nid] = QPointF(sx, sy)

        # Prune removed pipe bends
        current_pipe_ids = {p.id for p in network.pipes}
        for pid in list(self._edge_bends.keys()):
            if pid not in current_pipe_ids:
                del self._edge_bends[pid]

        # Draw pipes/edges
        if network.pipes:
            pipe_dict = {p.id: p for p in network.pipes}
            pair_counts = defaultdict(int)
            pair_index = defaultdict(int)

            for u, v, k, data in G.edges(keys=True, data=True):
                pair_key = tuple(sorted([u, v]))
                pair_counts[pair_key] += 1

            for u, v, k, data in G.edges(keys=True, data=True):
                pair_key = tuple(sorted([u, v]))
                count = pair_counts[pair_key]
                idx = pair_index[pair_key]
                pair_index[pair_key] += 1

                if count == 1:
                    rad = 0.0
                else:
                    step = 0.2
                    rad = ((idx + 1) // 2) * step * (1 if idx % 2 == 1 else -1)
                    if u > v:
                        rad = -rad

                p_obj = pipe_dict.get(data["id"])
                p_bend = self._edge_bends.get(data["id"], 0.0)

                p_item = PipeItem(
                    pipe_id=data["id"],
                    start_node_id=u,
                    end_node_id=v,
                    start_pos=scene_node_positions[u],
                    end_pos=scene_node_positions[v],
                    has_pump=data.get("has_pump", False),
                    label_text=data.get("label", data["id"]),
                    curvature=rad,
                    bend_ratio=p_bend,
                    graph_view=self,
                    valve_start=bool(p_obj.valve_start) if p_obj else False,
                    valve_end=bool(p_obj.valve_end) if p_obj else False,
                )

                # Pipe Tooltip
                if p_obj:
                    lines = [
                        f"ID: {p_obj.id}",
                        f"{p_obj.start} -> {p_obj.end}",
                        f"Length: {f'{p_obj.length} m' if p_obj.length is not None else 'not set'}",
                        f"Diameter: {f'{p_obj.diameter} m' if p_obj.diameter is not None else 'not set'}",
                        f"Roughness: {f'{p_obj.roughness} m' if p_obj.roughness is not None else 'not set'}",
                    ]
                    if p_obj.has_pump:
                        lines.append(f"Pump power: {p_obj.pump_power} kW")
                    p_item.setToolTip("\n".join(lines))

                p_item.set_highlighted(data["id"] in self._highlight_ids)
                p_item.set_selected(data["id"] == self._selected_pipe_id)
                self.scene.addItem(p_item)
                self._node_pipes[u].append(p_item)
                self._node_pipes[v].append(p_item)

        # Draw nodes
        for node in network.nodes:
            pos = scene_node_positions[node.id]
            n_item = NodeItem(node.id, node.type, pos, graph_view=self)

            # Node Tooltip
            lines = [
                f"ID: {node.id}",
                f"Type: {node.type.capitalize()}",
                f"Elevation: {f'{node.elevation} m' if node.elevation is not None else 'not set'}",
            ]
            if node.type.lower() == "reservoir":
                lines.append(f"Head: {f'{node.head} m' if node.head is not None else 'not set'}")
            n_item.setToolTip("\n".join(lines))

            n_item.set_highlighted(node.id in self._highlight_ids)
            n_item.set_selected(node.id == self._selected_node_id)
            self.scene.addItem(n_item)

        # Legend item
        self.legend_item = LegendItem()
        self.scene.addItem(self.legend_item)
        self.legend_item.setVisible(True)

        if self._pending_fit and self._is_viewport_valid():
            self.fit_in_view()
            self._pending_fit = False
        else:
            self.update_legend_position()

    @property
    def _needs_fit(self) -> bool:
        return self._pending_fit

    @_needs_fit.setter
    def _needs_fit(self, val: bool):
        self._pending_fit = val
