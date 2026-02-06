"""
Material Database Dialog

Dialog for managing and selecting materials.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QFrame, QAbstractItemView, QFormLayout,
    QDoubleSpinBox, QTextEdit
)

from menipy.gui import theme
from menipy.common.material_db import MaterialDB


class MaterialDialog(QDialog):
    """
    Dialog for viewing, editing, and selecting items from the database.
    Supports materials, needles, and syringes.
    
    Modes:
        - Manager: View/Edit/Delete items
    - Selector: Select an item and return it
    """
    
    item_selected = Signal(dict)
    
    # Alias for backward compatibility (users might expect material_selected)
    material_selected = item_selected
    
    def __init__(self, parent=None, selection_mode=False, table_type="materials"):
        super().__init__(parent)
        self.setWindowTitle(f"{table_type.title()} Database")
        self.resize(800, 500)
        self._selection_mode = selection_mode
        self._table_type = table_type
        self._db = MaterialDB()
        self._db.init_schema()
        
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setStyleSheet(theme.get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel(self._table_type.title())
        title.setProperty("title", True)
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        # Search
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(f"Search {self._table_type}...")
        self._search_input.textChanged.connect(self._filter_data)
        toolbar.addWidget(self._search_input)
        
        # Filter (only for materials)
        self._type_filter = QComboBox()
        if self._table_type == "materials":
            self._type_filter.addItems(["All Types", "Liquid", "Gas", "Solid"])
            self._type_filter.currentTextChanged.connect(self._filter_data)
            toolbar.addWidget(self._type_filter)
        else:
            self._type_filter.hide()
        
        toolbar.addStretch()
        
        # Actions
        add_btn = QPushButton(f"+ New {self._table_type[:-1].title()}")
        add_btn.clicked.connect(self._on_add)
        toolbar.addWidget(add_btn)
        
        layout.addLayout(toolbar)
        
        # Table configuration based on type
        self._table = QTableWidget()
        if self._table_type == "materials":
            headers = ["Name", "Type", "Density\n(kg/m³)", "Suff. Tens.\n(mN/m)", "Description"]
            stretch_col = 4
        elif self._table_type == "needles":
            headers = ["Name", "Gauge", "Outer Dia.\n(mm)", "Inner Dia.\n(mm)", "Description"]
            stretch_col = 4
        elif self._table_type == "syringes":
            headers = ["Name", "Volume\n(μL)", "Diameter\n(mm)", "Manufacturer", "Description"]
            stretch_col = 4
        else:
            headers = ["Name", "Description"]
            stretch_col = 1
            
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        
        if self._selection_mode:
            self._table.doubleClicked.connect(self._on_select)
        
        layout.addWidget(self._table)
        
        # Footer buttons
        footer = QHBoxLayout()
        footer.addStretch()
        
        if self._selection_mode:
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setProperty("secondary", True)
            cancel_btn.clicked.connect(self.reject)
            footer.addWidget(cancel_btn)
            
            select_btn = QPushButton("Select")
            select_btn.clicked.connect(self._on_select)
            footer.addWidget(select_btn)
        else:
            close_btn = QPushButton("Close")
            close_btn.setProperty("secondary", True)
            close_btn.clicked.connect(self.accept)
            footer.addWidget(close_btn)
        
        layout.addLayout(footer)
    
    def _load_data(self):
        """Load data from database."""
        if self._table_type == "materials":
            self._items = self._db.list_materials()
        elif self._table_type == "needles":
            self._items = self._db.list_needles()
        elif self._table_type == "syringes":
            self._items = self._db.list_syringes()
        else:
            self._items = []
        self._filter_data()
    
    def _filter_data(self):
        """Filter and display data in table."""
        search = self._search_input.text().lower()
        
        filtered = []
        for item in self._items:
            # Type filter (materials only)
            if self._table_type == "materials" and self._type_filter.isVisible():
                type_filter = self._type_filter.currentText().lower()
                if type_filter != "all types" and item.get("type") != type_filter:
                    continue
            
            # Simple search
            text = f"{item.get('name', '')} {item.get('description', '')}".lower()
            if search and search not in text:
                continue
                
            filtered.append(item)
        
        self._populate_table(filtered)
    
    def _populate_table(self, items: list):
        """Populate table widget."""
        self._table.setRowCount(0)
        for row, item in enumerate(items):
            self._table.insertRow(row)
            
            # Common: Name (0)
            name = QTableWidgetItem(str(item.get("name", "")))
            name.setData(Qt.ItemDataRole.UserRole, item)
            self._table.setItem(row, 0, name)
            
            # Specific columns
            if self._table_type == "materials":
                self._table.setItem(row, 1, QTableWidgetItem(str(item.get("type", ""))))
                
                dens = f"{item.get('density', 0):.1f}" if item.get("density") is not None else "-"
                self._table.setItem(row, 2, QTableWidgetItem(dens))
                
                st = f"{item.get('surface_tension', 0):.1f}" if item.get("surface_tension") is not None else "-"
                self._table.setItem(row, 3, QTableWidgetItem(st))
                
                self._table.setItem(row, 4, QTableWidgetItem(str(item.get("description", ""))))
                
            elif self._table_type == "needles":
                self._table.setItem(row, 1, QTableWidgetItem(str(item.get("gauge", ""))))
                
                od = f"{item.get('outer_diameter', 0):.2f}"
                self._table.setItem(row, 2, QTableWidgetItem(od))
                
                id_ = f"{item.get('inner_diameter', 0):.2f}" if item.get("inner_diameter") else "-"
                self._table.setItem(row, 3, QTableWidgetItem(id_))
                
                self._table.setItem(row, 4, QTableWidgetItem(str(item.get("description", ""))))
                
            elif self._table_type == "syringes":
                vol = f"{item.get('volume_ul', 0):.1f}"
                self._table.setItem(row, 1, QTableWidgetItem(vol))
                
                dia = f"{item.get('diameter_mm', 0):.2f}"
                self._table.setItem(row, 2, QTableWidgetItem(dia))
                
                self._table.setItem(row, 3, QTableWidgetItem(str(item.get("manufacturer", ""))))
                self._table.setItem(row, 4, QTableWidgetItem(str(item.get("description", ""))))

    def _on_select(self):
        """Handle selection."""
        row = self._table.currentRow()
        if row >= 0:
            item = self._table.item(row, 0)
            data = item.data(Qt.ItemDataRole.UserRole)
            self.item_selected.emit(data)
            self.accept()
            
    def _on_add(self):
        """Handle add button."""
        if self._table_type == "materials":
            dlg = _AddMaterialDialog(self)
            if dlg.exec():
                data = dlg.get_data()
                try:
                    name = data.pop("name")
                    self._db.upsert_material(name, data)
                    self._load_data()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save material: {e}")
        elif self._table_type == "needles":
            dlg = _AddNeedleDialog(self)
            if dlg.exec():
                data = dlg.get_data()
                try:
                    self._db.upsert_needle(**data)
                    self._load_data()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save needle: {e}")
        elif self._table_type == "syringes":
            dlg = _AddSyringeDialog(self)
            if dlg.exec():
                data = dlg.get_data()
                try:
                    self._db.upsert_syringe(**data)
                    self._load_data()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save syringe: {e}")
        else:
            QMessageBox.information(self, "Not Implemented", f"Adding new {self._table_type} is not implemented yet.")


class _AddMaterialDialog(QDialog):
    """Form to add/edit a material entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Material")
        self.resize(420, 340)
        self.setStyleSheet(theme.get_stylesheet())

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name = QLineEdit()
        self._type = QComboBox()
        self._type.addItems(["liquid", "gas", "solid"])

        self._density = QDoubleSpinBox()
        self._density.setRange(0.0, 50000.0)
        self._density.setDecimals(3)
        self._density.setSuffix(" kg/m³")
        self._density.setValue(1000.0)

        self._visc = QDoubleSpinBox()
        self._visc.setRange(0.0, 1e6)
        self._visc.setDecimals(3)
        self._visc.setSuffix(" mPa·s")

        self._st = QDoubleSpinBox()
        self._st.setRange(0.0, 2000.0)
        self._st.setDecimals(3)
        self._st.setSuffix(" mN/m")

        self._desc = QTextEdit()
        self._desc.setFixedHeight(80)

        form.addRow("Name*", self._name)
        form.addRow("Type", self._type)
        form.addRow("Density", self._density)
        form.addRow("Viscosity", self._visc)
        form.addRow("Surface Tension", self._st)
        form.addRow("Description", self._desc)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setProperty("secondary", True)
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Save")
        ok.clicked.connect(self._on_save)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

    def _on_save(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Validation", "Name is required.")
            return
        self.accept()

    def get_data(self) -> dict:
        """Get data.

        Returns
        -------
        type
        Description.
        """
        return {
            "name": self._name.text().strip(),
            "type": self._type.currentText(),
            "density": float(self._density.value()) if self._density.value() > 0 else None,
            "viscosity": float(self._visc.value()) if self._visc.value() > 0 else None,
            "surface_tension": float(self._st.value()) if self._st.value() > 0 else None,
            "description": self._desc.toPlainText().strip() or None,
        }


class _AddNeedleDialog(QDialog):
    """Form to add/edit a needle entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Needle")
        self.resize(380, 260)
        self.setStyleSheet(theme.get_stylesheet())

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name = QLineEdit()
        self._gauge = QLineEdit()

        self._od = QDoubleSpinBox()
        self._od.setRange(0.01, 100.0)
        self._od.setDecimals(3)
        self._od.setSuffix(" mm")
        self._od.setValue(0.8)

        self._id = QDoubleSpinBox()
        self._id.setRange(0.0, 100.0)
        self._id.setDecimals(3)
        self._id.setSuffix(" mm")

        self._desc = QTextEdit()
        self._desc.setFixedHeight(70)

        form.addRow("Name*", self._name)
        form.addRow("Gauge", self._gauge)
        form.addRow("Outer Diameter*", self._od)
        form.addRow("Inner Diameter", self._id)
        form.addRow("Description", self._desc)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setProperty("secondary", True)
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Save")
        ok.clicked.connect(self._on_save)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

    def _on_save(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Validation", "Name is required.")
            return
        if self._od.value() <= 0:
            QMessageBox.warning(self, "Validation", "Outer diameter must be > 0.")
            return
        self.accept()

    def get_data(self) -> dict:
        inner = float(self._id.value()) if self._id.value() > 0 else None
        return {
            "name": self._name.text().strip(),
            "gauge": self._gauge.text().strip() or None,
            "outer_diameter": float(self._od.value()),
            "inner_diameter": inner,
            "description": self._desc.toPlainText().strip() or None,
        }

class _AddSyringeDialog(QDialog):
    """Simple form to add a syringe entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Syringe")
        self.resize(380, 280)
        self.setStyleSheet(theme.get_stylesheet())

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name = QLineEdit()
        self._manufacturer = QLineEdit()

        self._volume = QDoubleSpinBox()
        self._volume.setRange(0.01, 1000000)
        self._volume.setDecimals(2)
        self._volume.setSuffix(" µL")
        self._volume.setValue(1000.0)

        self._diameter = QDoubleSpinBox()
        self._diameter.setRange(0.01, 100.0)
        self._diameter.setDecimals(3)
        self._diameter.setSuffix(" mm")
        self._diameter.setValue(4.7)

        self._desc = QTextEdit()
        self._desc.setFixedHeight(80)

        form.addRow("Name*", self._name)
        form.addRow("Manufacturer", self._manufacturer)
        form.addRow("Volume (µL)", self._volume)
        form.addRow("Diameter (mm)", self._diameter)
        form.addRow("Description", self._desc)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setProperty("secondary", True)
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Save")
        ok.clicked.connect(self._on_save)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

    def _on_save(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Validation", "Name is required.")
            return
        self.accept()

    def get_data(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "manufacturer": self._manufacturer.text().strip() or None,
            "volume_ul": float(self._volume.value()),
            "diameter_mm": float(self._diameter.value()),
            "description": self._desc.toPlainText().strip() or None,
        }
