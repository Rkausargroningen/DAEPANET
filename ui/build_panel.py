"""Left control panel for adding, editing, and deleting nodes and pipes."""

from typing import Optional, Dict, Any, List

from PySide6.QtCore import Signal, Qt, QLocale, QEvent
from PySide6.QtGui import QDoubleValidator, QPalette, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QPushButton, QCheckBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QLabel, QHeaderView, QLineEdit, QStackedWidget
)

from core.params import missing_parameters


def _parse_field_value(txt: str) -> Any:
    """Parse text from a QLineEdit into float or None. Returns raw string on ValueError for core validation."""
    s = txt.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return s


def _create_double_validator(parent=None) -> QDoubleValidator:
    """Create a QDoubleValidator accepting standard and scientific notation (e.g. 3.59e-4)."""
    v = QDoubleValidator(parent)
    v.setLocale(QLocale.c())
    v.setNotation(QDoubleValidator.ScientificNotation)
    return v


class BuildPanel(QWidget):
    node_added = Signal(str, dict)  # node_type, params_dict
    pipe_added = Signal(str, str, bool, float, dict)  # start_id, end_id, has_pump, pump_power, params_dict
    node_deleted = Signal(str)  # node_id
    pipe_deleted = Signal(str)  # pipe_id
    node_updated = Signal(str, dict)  # node_id, fields_dict
    pipe_updated = Signal(str, dict)  # pipe_id, fields_dict
    cleared = Signal()
    node_selected = Signal(str)  # node_id
    pipe_selected = Signal(str)  # pipe_id
    selection_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self._current_network = None
        self._selected_node_id: Optional[str] = None
        self._selected_pipe_id: Optional[str] = None
        self._updating_selection = False
        self._init_ui()
        self.update_theme_style()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 1. Add Node Group
        node_group = QGroupBox("Add Node")
        node_layout = QVBoxLayout(node_group)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.node_type_combo = QComboBox()
        self.node_type_combo.addItems(["Junction", "Tank", "Reservoir"])
        self.node_type_combo.currentTextChanged.connect(self._on_node_type_changed)
        type_row.addWidget(self.node_type_combo)
        node_layout.addLayout(type_row)

        node_form = QFormLayout()
        node_form.setContentsMargins(0, 0, 0, 0)
        self.node_elevation_input = QLineEdit()
        self.node_elevation_input.setPlaceholderText("not set")
        self.node_elevation_input.setValidator(_create_double_validator(self))

        self.node_head_input = QLineEdit()
        self.node_head_input.setPlaceholderText("not set")
        self.node_head_input.setValidator(_create_double_validator(self))
        self.node_head_input.setEnabled(False)  # Junction is initially selected

        node_form.addRow("Elevation (m):", self.node_elevation_input)
        node_form.addRow("Head (m):", self.node_head_input)
        node_layout.addLayout(node_form)

        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self._on_add_node)
        node_layout.addWidget(self.add_node_btn)
        layout.addWidget(node_group)

        # 2. Add Pipe Group
        pipe_group = QGroupBox("Add Pipe")
        pipe_layout = QVBoxLayout(pipe_group)

        combo_row = QHBoxLayout()
        self.start_combo = QComboBox()
        self.end_combo = QComboBox()
        combo_row.addWidget(QLabel("Start:"))
        combo_row.addWidget(self.start_combo)
        combo_row.addWidget(QLabel("End:"))
        combo_row.addWidget(self.end_combo)
        pipe_layout.addLayout(combo_row)

        pipe_form = QFormLayout()
        pipe_form.setContentsMargins(0, 0, 0, 0)
        self.pipe_length_input = QLineEdit()
        self.pipe_length_input.setPlaceholderText("not set")
        self.pipe_length_input.setValidator(_create_double_validator(self))

        self.pipe_diameter_input = QLineEdit()
        self.pipe_diameter_input.setPlaceholderText("not set")
        self.pipe_diameter_input.setValidator(_create_double_validator(self))

        self.pipe_roughness_input = QLineEdit()
        self.pipe_roughness_input.setPlaceholderText("not set")
        self.pipe_roughness_input.setValidator(_create_double_validator(self))

        pipe_form.addRow("Length (m):", self.pipe_length_input)
        pipe_form.addRow("Diameter (m):", self.pipe_diameter_input)
        pipe_form.addRow("Roughness (m):", self.pipe_roughness_input)
        pipe_layout.addLayout(pipe_form)

        pump_row = QHBoxLayout()
        self.pump_checkbox = QCheckBox("Contains Pump")
        self.pump_checkbox.setStyleSheet(
            "QCheckBox { font-weight: bold; } "
            "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; } "
            "QCheckBox::indicator:checked { background-color: #007ACC; border-color: #005999; }"
        )

        self.pump_power_spin = QDoubleSpinBox()
        self.pump_power_spin.setRange(0.0, 10000.0)
        self.pump_power_spin.setValue(10.0)
        self.pump_power_spin.setSuffix(" kW")
        self.pump_power_spin.setEnabled(False)
        self.pump_power_spin.setToolTip("stored only, not used in the matrices yet")
        self.pump_checkbox.toggled.connect(self.pump_power_spin.setEnabled)

        pump_row.addWidget(self.pump_checkbox)
        pump_row.addWidget(self.pump_power_spin)
        pipe_layout.addLayout(pump_row)

        self.add_pipe_btn = QPushButton("Add Pipe")
        self.add_pipe_btn.clicked.connect(self._on_add_pipe)
        pipe_layout.addWidget(self.add_pipe_btn)
        layout.addWidget(pipe_group)

        # 3. Nodes Table
        nodes_group = QGroupBox("Nodes")
        nodes_layout = QVBoxLayout(nodes_group)
        self.nodes_table = QTableWidget(0, 2)
        self.nodes_table.setHorizontalHeaderLabels(["ID", "Type"])
        self.nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nodes_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.nodes_table.setSelectionMode(QTableWidget.SingleSelection)
        self.nodes_table.itemSelectionChanged.connect(self._on_node_selection_changed)
        nodes_layout.addWidget(self.nodes_table)

        self.del_node_btn = QPushButton("Delete Selected Node")
        self.del_node_btn.clicked.connect(self._on_delete_node)
        nodes_layout.addWidget(self.del_node_btn)
        layout.addWidget(nodes_group)

        # 4. Pipes Table
        pipes_group = QGroupBox("Pipes")
        pipes_layout = QVBoxLayout(pipes_group)
        self.pipes_table = QTableWidget(0, 4)
        self.pipes_table.setHorizontalHeaderLabels(["ID", "Start", "End", "Pump?"])
        self.pipes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pipes_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pipes_table.setSelectionMode(QTableWidget.SingleSelection)
        self.pipes_table.itemSelectionChanged.connect(self._on_pipe_selection_changed)
        pipes_layout.addWidget(self.pipes_table)

        self.del_pipe_btn = QPushButton("Delete Selected Pipe")
        self.del_pipe_btn.clicked.connect(self._on_delete_pipe)
        pipes_layout.addWidget(self.del_pipe_btn)
        layout.addWidget(pipes_group)

        # 5. Properties Editor
        props_group = QGroupBox("Properties")
        props_layout = QVBoxLayout(props_group)

        self.props_stack = QStackedWidget()

        # Page 0: Hint
        self.props_hint_widget = QWidget()
        hint_layout = QVBoxLayout(self.props_hint_widget)
        self.props_hint_label = QLabel("Select a node or pipe above to view and edit its properties.")
        self.props_hint_label.setWordWrap(True)
        self.props_hint_label.setStyleSheet("color: gray; font-style: italic;")
        hint_layout.addWidget(self.props_hint_label)
        hint_layout.addStretch()
        self.props_stack.addWidget(self.props_hint_widget)

        # Page 1: Node Properties
        self.props_node_widget = QWidget()
        node_props_layout = QFormLayout(self.props_node_widget)
        node_props_layout.setContentsMargins(0, 0, 0, 0)
        self.prop_node_id_lbl = QLabel()
        self.prop_node_type_lbl = QLabel()
        self.prop_node_elev_edit = QLineEdit()
        self.prop_node_elev_edit.setPlaceholderText("not set")
        self.prop_node_elev_edit.setValidator(_create_double_validator(self))
        self.prop_node_head_edit = QLineEdit()
        self.prop_node_head_edit.setPlaceholderText("not set")
        self.prop_node_head_edit.setValidator(_create_double_validator(self))
        self.prop_node_cap_edit = QLineEdit()
        self.prop_node_cap_edit.setPlaceholderText("not set")
        self.prop_node_cap_edit.setValidator(_create_double_validator(self))
        self.prop_node_cap_edit.setToolTip("stored for the planned comparison with EPANET")

        node_props_layout.addRow("ID:", self.prop_node_id_lbl)
        node_props_layout.addRow("Type:", self.prop_node_type_lbl)
        node_props_layout.addRow("Elevation (m):", self.prop_node_elev_edit)
        node_props_layout.addRow("Head (m):", self.prop_node_head_edit)
        node_props_layout.addRow("Capacity:", self.prop_node_cap_edit)

        node_btn_row = QHBoxLayout()
        self.apply_node_btn = QPushButton("Apply")
        self.revert_node_btn = QPushButton("Revert")
        node_btn_row.addWidget(self.apply_node_btn)
        node_btn_row.addWidget(self.revert_node_btn)
        node_props_layout.addRow(node_btn_row)

        self.prop_node_elev_edit.returnPressed.connect(self._on_apply_node)
        self.prop_node_head_edit.returnPressed.connect(self._on_apply_node)
        self.prop_node_cap_edit.returnPressed.connect(self._on_apply_node)
        self.apply_node_btn.clicked.connect(self._on_apply_node)
        self.revert_node_btn.clicked.connect(self._on_revert_node)

        self.props_stack.addWidget(self.props_node_widget)

        # Page 2: Pipe Properties
        self.props_pipe_widget = QWidget()
        pipe_props_layout = QFormLayout(self.props_pipe_widget)
        pipe_props_layout.setContentsMargins(0, 0, 0, 0)
        self.prop_pipe_id_lbl = QLabel()
        self.prop_pipe_conn_lbl = QLabel()
        self.prop_pipe_length_edit = QLineEdit()
        self.prop_pipe_length_edit.setPlaceholderText("not set")
        self.prop_pipe_length_edit.setValidator(_create_double_validator(self))
        self.prop_pipe_diam_edit = QLineEdit()
        self.prop_pipe_diam_edit.setPlaceholderText("not set")
        self.prop_pipe_diam_edit.setValidator(_create_double_validator(self))
        self.prop_pipe_rough_edit = QLineEdit()
        self.prop_pipe_rough_edit.setPlaceholderText("not set")
        self.prop_pipe_rough_edit.setValidator(_create_double_validator(self))
        self.prop_pipe_power_edit = QLineEdit()
        self.prop_pipe_power_edit.setPlaceholderText("not set")
        self.prop_pipe_power_edit.setValidator(_create_double_validator(self))
        self.prop_pipe_power_edit.setToolTip("stored for the planned comparison with EPANET")

        pipe_props_layout.addRow("ID:", self.prop_pipe_id_lbl)
        pipe_props_layout.addRow("Connection:", self.prop_pipe_conn_lbl)
        pipe_props_layout.addRow("Length (m):", self.prop_pipe_length_edit)
        pipe_props_layout.addRow("Diameter (m):", self.prop_pipe_diam_edit)
        pipe_props_layout.addRow("Roughness (m):", self.prop_pipe_rough_edit)
        pipe_props_layout.addRow("Pump Power (kW):", self.prop_pipe_power_edit)

        pipe_btn_row = QHBoxLayout()
        self.apply_pipe_btn = QPushButton("Apply")
        self.revert_pipe_btn = QPushButton("Revert")
        pipe_btn_row.addWidget(self.apply_pipe_btn)
        pipe_btn_row.addWidget(self.revert_pipe_btn)
        pipe_props_layout.addRow(pipe_btn_row)

        self.prop_pipe_length_edit.returnPressed.connect(self._on_apply_pipe)
        self.prop_pipe_diam_edit.returnPressed.connect(self._on_apply_pipe)
        self.prop_pipe_rough_edit.returnPressed.connect(self._on_apply_pipe)
        self.prop_pipe_power_edit.returnPressed.connect(self._on_apply_pipe)
        self.apply_pipe_btn.clicked.connect(self._on_apply_pipe)
        self.revert_pipe_btn.clicked.connect(self._on_revert_pipe)

        self.props_stack.addWidget(self.props_pipe_widget)

        props_layout.addWidget(self.props_stack)
        layout.addWidget(props_group)

        # 6. Missing-Parameters Indicator
        param_status_layout = QHBoxLayout()
        self.params_status_label = QLabel("")
        self.params_details_btn = QPushButton("Details")
        self.params_details_btn.setFixedWidth(70)
        self.params_details_btn.clicked.connect(self._on_show_missing_details)
        self.params_details_btn.setVisible(False)
        param_status_layout.addWidget(self.params_status_label)
        param_status_layout.addWidget(self.params_details_btn)
        layout.addLayout(param_status_layout)

        # 7. Clear All Button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.clear_btn.clicked.connect(self._on_clear_all)
        layout.addWidget(self.clear_btn)

    def _on_node_type_changed(self, text: str):
        is_res = text.strip().lower() == "reservoir"
        self.node_head_input.setEnabled(is_res)
        if not is_res:
            self.node_head_input.clear()

    def _on_add_node(self):
        ntype = self.node_type_combo.currentText().lower()
        params: Dict[str, Any] = {
            "elevation": _parse_field_value(self.node_elevation_input.text())
        }
        if ntype == "reservoir":
            params["head"] = _parse_field_value(self.node_head_input.text())
        self.node_added.emit(ntype, params)

    def _on_add_pipe(self):
        start = self.start_combo.currentText()
        end = self.end_combo.currentText()
        if not start or not end:
            QMessageBox.warning(self, "Invalid Selection", "Please select both a start and end node.")
            return
        if start == end:
            QMessageBox.warning(self, "Invalid Pipe", "Start and End node cannot be the same (self-loops not allowed).")
            return
        has_pump = self.pump_checkbox.isChecked()
        power = self.pump_power_spin.value() if has_pump else 0.0

        params: Dict[str, Any] = {
            "length": _parse_field_value(self.pipe_length_input.text()),
            "diameter": _parse_field_value(self.pipe_diameter_input.text()),
            "roughness": _parse_field_value(self.pipe_roughness_input.text()),
        }

        self.pipe_added.emit(start, end, has_pump, power, params)

    def _on_node_selection_changed(self):
        if self._updating_selection:
            return
        if self.nodes_table.selectedItems():
            row = self.nodes_table.currentRow()
            if row >= 0 and self.nodes_table.item(row, 0):
                node_id = self.nodes_table.item(row, 0).text()
                self._selected_node_id = node_id
                self._selected_pipe_id = None
                self._updating_selection = True
                self.pipes_table.clearSelection()
                self._updating_selection = False
                if self._current_network:
                    node = next((n for n in self._current_network.nodes if n.id == node_id), None)
                    if node:
                        self._populate_node_properties(node)
                        self.props_stack.setCurrentWidget(self.props_node_widget)
                        self.node_selected.emit(node_id)
                        return
        self._selected_node_id = None
        if not self._selected_pipe_id:
            self.props_stack.setCurrentWidget(self.props_hint_widget)
            self.selection_cleared.emit()

    def _on_pipe_selection_changed(self):
        if self._updating_selection:
            return
        if self.pipes_table.selectedItems():
            row = self.pipes_table.currentRow()
            if row >= 0 and self.pipes_table.item(row, 0):
                pipe_id = self.pipes_table.item(row, 0).text()
                self._selected_pipe_id = pipe_id
                self._selected_node_id = None
                self._updating_selection = True
                self.nodes_table.clearSelection()
                self._updating_selection = False
                if self._current_network:
                    pipe = next((p for p in self._current_network.pipes if p.id == pipe_id), None)
                    if pipe:
                        self._populate_pipe_properties(pipe)
                        self.props_stack.setCurrentWidget(self.props_pipe_widget)
                        self.pipe_selected.emit(pipe_id)
                        return
        self._selected_pipe_id = None
        if not self._selected_node_id:
            self.props_stack.setCurrentWidget(self.props_hint_widget)
            self.selection_cleared.emit()

    def select_node(self, node_id: str):
        """Programmatically select a node row in the nodes table and display its properties."""
        self._updating_selection = True
        try:
            self.pipes_table.clearSelection()
            self._selected_pipe_id = None
            found_row = -1
            for r in range(self.nodes_table.rowCount()):
                it = self.nodes_table.item(r, 0)
                if it and it.text() == node_id:
                    found_row = r
                    break
            if found_row >= 0:
                self.nodes_table.selectRow(found_row)
                self._selected_node_id = node_id
                if self._current_network:
                    node = next((n for n in self._current_network.nodes if n.id == node_id), None)
                    if node:
                        self._populate_node_properties(node)
                        self.props_stack.setCurrentWidget(self.props_node_widget)
                self.node_selected.emit(node_id)
            else:
                self.nodes_table.clearSelection()
                self._selected_node_id = None
                self.props_stack.setCurrentWidget(self.props_hint_widget)
                self.selection_cleared.emit()
        finally:
            self._updating_selection = False

    def select_pipe(self, pipe_id: str):
        """Programmatically select a pipe row in the pipes table and display its properties."""
        self._updating_selection = True
        try:
            self.nodes_table.clearSelection()
            self._selected_node_id = None
            found_row = -1
            for r in range(self.pipes_table.rowCount()):
                it = self.pipes_table.item(r, 0)
                if it and it.text() == pipe_id:
                    found_row = r
                    break
            if found_row >= 0:
                self.pipes_table.selectRow(found_row)
                self._selected_pipe_id = pipe_id
                if self._current_network:
                    pipe = next((p for p in self._current_network.pipes if p.id == pipe_id), None)
                    if pipe:
                        self._populate_pipe_properties(pipe)
                        self.props_stack.setCurrentWidget(self.props_pipe_widget)
                self.pipe_selected.emit(pipe_id)
            else:
                self.pipes_table.clearSelection()
                self._selected_pipe_id = None
                self.props_stack.setCurrentWidget(self.props_hint_widget)
                self.selection_cleared.emit()
        finally:
            self._updating_selection = False

    def clear_selection(self):
        """Clear all table selections and reset properties panel to hint."""
        self._updating_selection = True
        try:
            self.nodes_table.clearSelection()
            self.pipes_table.clearSelection()
            self._selected_node_id = None
            self._selected_pipe_id = None
            self.props_stack.setCurrentWidget(self.props_hint_widget)
            self.selection_cleared.emit()
        finally:
            self._updating_selection = False

    def _populate_node_properties(self, node):
        self.prop_node_id_lbl.setText(node.id)
        self.prop_node_type_lbl.setText(node.type.capitalize())
        self.prop_node_elev_edit.setText("" if node.elevation is None else f"{node.elevation:g}")

        is_res = node.type == "reservoir"
        self.prop_node_head_edit.setEnabled(is_res)
        self.prop_node_head_edit.setText("" if node.head is None else f"{node.head:g}")

        is_tank = node.type == "tank"
        self.prop_node_cap_edit.setEnabled(is_tank)
        self.prop_node_cap_edit.setText("" if node.capacity is None else f"{node.capacity:g}")

    def _populate_pipe_properties(self, pipe):
        self.prop_pipe_id_lbl.setText(pipe.id)
        self.prop_pipe_conn_lbl.setText(f"{pipe.start} -> {pipe.end}")
        self.prop_pipe_length_edit.setText("" if pipe.length is None else f"{pipe.length:g}")
        self.prop_pipe_diam_edit.setText("" if pipe.diameter is None else f"{pipe.diameter:g}")
        self.prop_pipe_rough_edit.setText("" if pipe.roughness is None else f"{pipe.roughness:g}")

        self.prop_pipe_power_edit.setEnabled(pipe.has_pump)
        self.prop_pipe_power_edit.setText("" if pipe.pump_power is None else f"{pipe.pump_power:g}")

    def _on_apply_node(self):
        if not self._selected_node_id or not self._current_network:
            return
        node = next((n for n in self._current_network.nodes if n.id == self._selected_node_id), None)
        if not node:
            return
        fields: Dict[str, Any] = {}
        fields["elevation"] = _parse_field_value(self.prop_node_elev_edit.text())
        if node.type == "reservoir":
            fields["head"] = _parse_field_value(self.prop_node_head_edit.text())
        if node.type == "tank":
            fields["capacity"] = _parse_field_value(self.prop_node_cap_edit.text())
        self.node_updated.emit(self._selected_node_id, fields)

    def _on_revert_node(self):
        if not self._selected_node_id or not self._current_network:
            return
        node = next((n for n in self._current_network.nodes if n.id == self._selected_node_id), None)
        if node:
            self._populate_node_properties(node)

    def _on_apply_pipe(self):
        if not self._selected_pipe_id or not self._current_network:
            return
        pipe = next((p for p in self._current_network.pipes if p.id == self._selected_pipe_id), None)
        if not pipe:
            return
        fields: Dict[str, Any] = {}
        fields["length"] = _parse_field_value(self.prop_pipe_length_edit.text())
        fields["diameter"] = _parse_field_value(self.prop_pipe_diam_edit.text())
        fields["roughness"] = _parse_field_value(self.prop_pipe_rough_edit.text())
        if pipe.has_pump:
            fields["pump_power"] = _parse_field_value(self.prop_pipe_power_edit.text())
        self.pipe_updated.emit(self._selected_pipe_id, fields)

    def _on_revert_pipe(self):
        if not self._selected_pipe_id or not self._current_network:
            return
        pipe = next((p for p in self._current_network.pipes if p.id == self._selected_pipe_id), None)
        if pipe:
            self._populate_pipe_properties(pipe)

    def _on_show_missing_details(self):
        if not self._current_network:
            return
        missing = missing_parameters(self._current_network)
        if not missing:
            msg = "All required physical parameters are set."
        else:
            msg = "\n".join(f"• {m}" for m in missing)
        QMessageBox.information(self, "Missing Parameters Details", msg)

    def update_network(self, network):
        """Update node/pipe dropdowns, tables, properties editor, and parameter indicator."""
        self._current_network = network
        prev_node_id = self._selected_node_id
        prev_pipe_id = self._selected_pipe_id

        node_ids = [n.id for n in network.nodes]
        self.start_combo.blockSignals(True)
        self.end_combo.blockSignals(True)
        self.start_combo.clear()
        self.end_combo.clear()
        self.start_combo.addItems(node_ids)
        self.end_combo.addItems(node_ids)

        if len(node_ids) >= 2:
            self.start_combo.setCurrentIndex(0)
            self.end_combo.setCurrentIndex(1)
        elif len(node_ids) == 1:
            self.start_combo.setCurrentIndex(0)
            self.end_combo.setCurrentIndex(0)

        self.start_combo.blockSignals(False)
        self.end_combo.blockSignals(False)

        self._updating_selection = True

        # Nodes table
        self.nodes_table.setRowCount(len(network.nodes))
        for r, node in enumerate(network.nodes):
            self.nodes_table.setItem(r, 0, QTableWidgetItem(node.id))
            self.nodes_table.setItem(r, 1, QTableWidgetItem(node.type.capitalize()))

        # Pipes table
        self.pipes_table.setRowCount(len(network.pipes))
        for r, pipe in enumerate(network.pipes):
            pump_str = f"Yes ({pipe.pump_power:.1f} kW)" if pipe.has_pump else "No"
            self.pipes_table.setItem(r, 0, QTableWidgetItem(pipe.id))
            self.pipes_table.setItem(r, 1, QTableWidgetItem(pipe.start))
            self.pipes_table.setItem(r, 2, QTableWidgetItem(pipe.end))
            self.pipes_table.setItem(r, 3, QTableWidgetItem(pump_str))

        # Restore selection by ID
        restored = False
        if prev_node_id:
            for r in range(self.nodes_table.rowCount()):
                if self.nodes_table.item(r, 0).text() == prev_node_id:
                    self.nodes_table.selectRow(r)
                    node = next((n for n in network.nodes if n.id == prev_node_id), None)
                    if node:
                        self._populate_node_properties(node)
                        self.props_stack.setCurrentWidget(self.props_node_widget)
                        self._selected_node_id = prev_node_id
                        self._selected_pipe_id = None
                        restored = True
                    break
        if not restored and prev_pipe_id:
            for r in range(self.pipes_table.rowCount()):
                if self.pipes_table.item(r, 0).text() == prev_pipe_id:
                    self.pipes_table.selectRow(r)
                    pipe = next((p for p in network.pipes if p.id == prev_pipe_id), None)
                    if pipe:
                        self._populate_pipe_properties(pipe)
                        self.props_stack.setCurrentWidget(self.props_pipe_widget)
                        self._selected_pipe_id = prev_pipe_id
                        self._selected_node_id = None
                        restored = True
                    break

        if not restored:
            self._selected_node_id = None
            self._selected_pipe_id = None
            self.nodes_table.clearSelection()
            self.pipes_table.clearSelection()
            self.props_stack.setCurrentWidget(self.props_hint_widget)

        self._updating_selection = False

        # Update Missing Parameters indicator
        if not network.nodes and not network.pipes:
            self.params_status_label.setText("")
            self.params_details_btn.setVisible(False)
        else:
            missing = missing_parameters(network)
            if not missing:
                self.params_status_label.setText("Parameters: complete")
                self.params_status_label.setStyleSheet("color: green; font-weight: bold;")
                self.params_details_btn.setVisible(False)
            else:
                self.params_status_label.setText(f"Parameters: {len(missing)} missing")
                self.params_status_label.setStyleSheet("color: #d97706; font-weight: bold;")
                self.params_details_btn.setVisible(True)

    def _on_delete_node(self):
        row = self.nodes_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a node to delete.")
            return
        node_id = self.nodes_table.item(row, 0).text()
        attached_pipes = []
        if self._current_network is not None:
            attached_pipes = [p for p in self._current_network.pipes if p.start == node_id or p.end == node_id]
        pipe_count = len(attached_pipes)

        msg = f"Are you sure you want to delete node {node_id}?"
        if pipe_count > 0:
            msg += f"\nThis will also delete {pipe_count} pipe(s) attached to it."

        res = QMessageBox.question(self, "Confirm Delete Node", msg, QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self._selected_node_id = None
            self.node_deleted.emit(node_id)

    def _on_delete_pipe(self):
        row = self.pipes_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a pipe to delete.")
            return
        pipe_id = self.pipes_table.item(row, 0).text()
        res = QMessageBox.question(self, "Confirm Delete Pipe", f"Are you sure you want to delete pipe {pipe_id}?", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self._selected_pipe_id = None
            self.pipe_deleted.emit(pipe_id)

    def _on_clear_all(self):
        res = QMessageBox.question(self, "Confirm Clear All", "Are you sure you want to clear all nodes and pipes?", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self._selected_node_id = None
            self._selected_pipe_id = None
            self.cleared.emit()

    def update_theme_style(self, is_dark: Optional[bool] = None):
        """Ensure high contrast text and background for all controls in both light and dark themes."""
        if getattr(self, "_updating_theme", False):
            return
        self._updating_theme = True
        try:
            if is_dark is None:
                is_dark = self.palette().color(QPalette.Window).value() < 128

            if is_dark:
                ph_color = QColor("#888888")
                self.setStyleSheet("""
                    BuildPanel { background: transparent; }
                    QGroupBox {
                        font-weight: bold;
                        color: #E0E0E0;
                        border: 1px solid #444444;
                        border-radius: 4px;
                        margin-top: 8px;
                        padding-top: 10px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px;
                    }
                    QLabel { color: #E0E0E0; }
                    QPushButton {
                        background-color: #383838;
                        color: #F0F0F0;
                        border: 1px solid #505050;
                        border-radius: 4px;
                        padding: 4px 10px;
                        font-weight: 500;
                    }
                    QPushButton:hover { background-color: #484848; }
                    QPushButton:pressed { background-color: #2D2D2D; }
                    QPushButton:disabled {
                        background-color: #2A2A2A;
                        color: #666666;
                        border: 1px solid #3A3A3A;
                    }
                    QComboBox {
                        background-color: #252525;
                        color: #F0F0F0;
                        border: 1px solid #505050;
                        border-radius: 4px;
                        padding: 3px 6px;
                    }
                    QComboBox:disabled {
                        background-color: #2A2A2A;
                        color: #666666;
                    }
                    QLineEdit {
                        background-color: #252525;
                        color: #F0F0F0;
                        border: 1px solid #505050;
                        border-radius: 4px;
                        padding: 3px 6px;
                    }
                    QLineEdit:disabled {
                        background-color: #2A2A2A;
                        color: #666666;
                    }
                """)
            else:
                ph_color = QColor("#5A626A")
                self.setStyleSheet("""
                    BuildPanel { background: transparent; }
                    QGroupBox {
                        font-weight: bold;
                        color: #1E1E1E;
                        border: 1px solid #D0D4DC;
                        border-radius: 4px;
                        margin-top: 8px;
                        padding-top: 10px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px;
                    }
                    QLabel { color: #1E1E1E; }
                    QPushButton {
                        background-color: #E4E7EB;
                        color: #1E1E1E;
                        border: 1px solid #B8BDC5;
                        border-radius: 4px;
                        padding: 4px 10px;
                        font-weight: 500;
                    }
                    QPushButton:hover { background-color: #D8DCE2; }
                    QPushButton:pressed { background-color: #CCD1D9; }
                    QPushButton:disabled {
                        background-color: #EAECEF;
                        color: #8C939D;
                        border: 1px solid #D5D9E0;
                    }
                    QComboBox {
                        background-color: #FFFFFF;
                        color: #1E1E1E;
                        border: 1px solid #B8BDC5;
                        border-radius: 4px;
                        padding: 3px 6px;
                    }
                    QComboBox:disabled {
                        background-color: #F4F5F7;
                        color: #8C939D;
                    }
                    QLineEdit {
                        background-color: #FFFFFF;
                        color: #1E1E1E;
                        border: 1px solid #B8BDC5;
                        border-radius: 4px;
                        padding: 3px 6px;
                    }
                    QLineEdit:disabled {
                        background-color: #F4F5F7;
                        color: #8C939D;
                    }
                """)

            for w in self.findChildren(QLineEdit):
                pal = w.palette()
                pal.setColor(QPalette.PlaceholderText, ph_color)
                w.setPalette(pal)
        finally:
            self._updating_theme = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange and not getattr(self, "_updating_theme", False):
            self.update_theme_style()
