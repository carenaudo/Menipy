"""
Experiment Selector View

The initial screen where users select their experiment type.
Displays experiment type cards and recent projects.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QListWidget,
    QListWidgetItem, QMenu
)

from menipy.gui import theme
from menipy.gui.widgets.experiment_card import (
    ExperimentCard, EXPERIMENT_DEFINITIONS, create_experiment_card
)


class RecentProjectItem(QListWidgetItem):
    """List item representing a recent project."""
    
    def __init__(self, filename: str, experiment_type: str, date_str: str):
        super().__init__()
        self.filename = filename
        self.experiment_type = experiment_type
        self.date_str = date_str
        
        # Get icon for experiment type
        icon = "💧"  # default
        for defn in EXPERIMENT_DEFINITIONS:
            if defn["type"] == experiment_type:
                icon = defn["icon"]
                break
        
        self.setText(f"{icon}  {filename}    {date_str}")


class RecentProjectsPanel(QFrame):
    """
    Panel displaying recent projects with quick access.
    """
    
    project_selected = Signal(str)  # Emits file path
    open_dialog_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the recent projects panel UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_SECONDARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header row
        header_layout = QHBoxLayout()
        
        header_label = QLabel("📂 Recent Projects")
        header_font = QFont(theme.FONT_FAMILY, theme.FONT_SIZE_LARGE)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        open_button = QPushButton("Open Project…")
        open_button.setProperty("secondary", True)
        open_button.clicked.connect(self.open_dialog_requested.emit)
        header_layout.addWidget(open_button)
        
        layout.addLayout(header_layout)
        
        # Projects list
        self.projects_list = QListWidget()
        self.projects_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {theme.BG_TERTIARY};
                border: 1px solid {theme.BORDER_DEFAULT};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {theme.BORDER_DEFAULT};
            }}
            QListWidget::item:hover {{
                background-color: {theme.BG_HOVER};
            }}
            QListWidget::item:selected {{
                background-color: {theme.ACCENT_BLUE};
            }}
        """)
        self.projects_list.setFixedHeight(120)
        self.projects_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.projects_list.customContextMenuRequested.connect(self._show_context_menu)
        self.projects_list.itemDoubleClicked.connect(self._on_project_double_clicked)
        layout.addWidget(self.projects_list)
    
    def set_recent_projects(self, projects: list[dict]):
        """
        Set the list of recent projects.
        
        Args:
            projects: List of dicts with keys: filename, experiment_type, date_str, path
        """
        self.projects_list.clear()
        for proj in projects:
            item = RecentProjectItem(
                filename=proj.get("filename", "Unknown"),
                experiment_type=proj.get("experiment_type", theme.EXPERIMENT_SESSILE),
                date_str=proj.get("date_str", "")
            )
            item.setData(Qt.ItemDataRole.UserRole, proj.get("path", ""))
            self.projects_list.addItem(item)
    
    def _on_project_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on a project."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.project_selected.emit(path)
    
    def _show_context_menu(self, position):
        """Show context menu for project item."""
        item = self.projects_list.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme.BG_SECONDARY};
                color: {theme.TEXT_PRIMARY};
                border: 1px solid {theme.BORDER_DEFAULT};
            }}
            QMenu::item {{
                padding: 8px 24px;
            }}
            QMenu::item:selected {{
                background-color: {theme.ACCENT_BLUE};
            }}
        """)
        
        open_action = menu.addAction("Open")
        menu.addSeparator()
        delete_action = menu.addAction("Remove from List")
        show_folder_action = menu.addAction("Show in Folder")
        
        action = menu.exec_(self.projects_list.mapToGlobal(position))
        
        if action == open_action:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.project_selected.emit(path)
        elif action == delete_action:
            row = self.projects_list.row(item)
            self.projects_list.takeItem(row)
        elif action == show_folder_action:
            # TODO: Open file explorer at location
            pass


class ExperimentSelectorView(QWidget):
    """
    Main experiment selector screen.
    
    Displays a grid of experiment type cards and recent projects.
    Users select their experiment type before proceeding to the main analysis window.
    
    Signals:
        experiment_selected: Emitted when user selects an experiment type.
        project_opened: Emitted when user opens a recent project.
    """
    
    experiment_selected = Signal(str)
    project_opened = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("experimentSelector")
        self._cards: list[ExperimentCard] = []
        self._current_focus_index = 0
        self._setup_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def _setup_ui(self):
        """Set up the experiment selector UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(24)
        
        # Header
        header_label = QLabel("Select Your Experiment Type")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setProperty("title", True)
        header_font = QFont(theme.FONT_FAMILY, theme.FONT_SIZE_TITLE)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
        main_layout.addWidget(header_label)
        
        # Scroll area for cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Cards grid
        cards_container = QWidget()
        cards_layout = QGridLayout(cards_container)
        cards_layout.setSpacing(theme.CARD_SPACING)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create cards for each experiment type
        row = 0
        col = 0
        max_cols = 3
        
        for defn in EXPERIMENT_DEFINITIONS:
            card = ExperimentCard(
                experiment_type=defn["type"],
                title=defn["title"],
                description=defn["description"],
                icon=defn["icon"],
                parent=self
            )
            card.selected.connect(self._on_experiment_selected)
            cards_layout.addWidget(card, row, col)
            self._cards.append(card)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        scroll_layout.addWidget(cards_container, alignment=Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {theme.BORDER_DEFAULT};")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)
        
        # Recent projects panel
        self.recent_projects_panel = RecentProjectsPanel()
        self.recent_projects_panel.project_selected.connect(self._on_project_selected)
        self.recent_projects_panel.open_dialog_requested.connect(
            lambda: self.project_opened.emit("")
        )
        main_layout.addWidget(self.recent_projects_panel)
    
    def _on_experiment_selected(self, experiment_type: str):
        """Handle experiment type selection."""
        self.experiment_selected.emit(experiment_type)
    
    def _on_project_selected(self, path: str):
        """Handle recent project selection."""
        self.project_opened.emit(path)
    
    def set_recent_projects(self, projects: list[dict]):
        """
        Set the list of recent projects to display.
        
        Args:
            projects: List of project dicts with filename, experiment_type, date_str, path
        """
        self.recent_projects_panel.set_recent_projects(projects)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation."""
        key = event.key()
        
        if key == Qt.Key.Key_Tab:
            # Navigate between cards
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._current_focus_index = (self._current_focus_index - 1) % len(self._cards)
            else:
                self._current_focus_index = (self._current_focus_index + 1) % len(self._cards)
            self._cards[self._current_focus_index].setFocus()
            event.accept()
        
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # Select current card
            if 0 <= self._current_focus_index < len(self._cards):
                card = self._cards[self._current_focus_index]
                card.selected.emit(card.experiment_type)
            event.accept()
        
        elif key == Qt.Key.Key_Escape:
            # Could emit close signal or show confirmation
            event.accept()
        
        else:
            super().keyPressEvent(event)
    
    def showEvent(self, event):
        """Handle show event - set initial focus."""
        super().showEvent(event)
        if self._cards:
            self._cards[0].setFocus()
            self._current_focus_index = 0
