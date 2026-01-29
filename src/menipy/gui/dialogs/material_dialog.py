"""
Material Database Dialog

Dialog for managing and selecting materials.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QFrame, QAbstractItemView
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
        QMessageBox.information(self, "Not Implemented", f"Adding new {self._table_type} is not implemented yet.")
