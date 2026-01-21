"""Plugin dock and menu controller for the main window."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QDockWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QDialog,
)

try:
    from menipy.common.plugins import PluginDB, discover_into_db, load_active_plugins
except Exception:
    PluginDB = None  # type: ignore[assignment]

    def discover_into_db(*_args, **_kwargs):  # type: ignore[unused-argument]
        return None

    def load_active_plugins(*_args, **_kwargs):  # type: ignore[unused-argument]
        return None


try:
    from menipy.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
    from menipy.gui.viewmodels.plugins_vm import PluginsViewModel
except Exception:
    PluginManagerDialog = None  # type: ignore[assignment]
    PluginsViewModel = None  # type: ignore[assignment]


class PluginsController:
    """Encapsulates plugin discovery UI wiring for ``MainWindow``."""

    def __init__(self, window: QMainWindow, settings: Any) -> None:
        self.window = window
        self.settings = settings
        self.db: Optional[PluginDB] = None  # type: ignore[assignment]
        self.dock: Optional[QDockWidget] = None
        self.action_toggle: Optional[QAction] = None
        self.action_manager: Optional[QAction] = None
        self.action_add_folder: Optional[QAction] = None

        self._build_plugin_dock()
        self._add_toggle_to_view_menu()
        self._ensure_plugins_menu()

    # -------------------------- setup helpers --------------------------

    def _build_plugin_dock(self) -> None:
        if not PluginDB:
            return
        try:
            self.db = PluginDB()
            self.db.init_schema()
            self._plugins_discover(self.settings.plugin_dirs or [])
            load_active_plugins(self.db)

            dock = QDockWidget("Plugins", self.window)
            dock.setObjectName("pluginDock")
            dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            info = QLabel("Plugins loaded.", dock)
            info.setMargin(8)
            dock.setWidget(info)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.hide()
            self.dock = dock
        except Exception as exc:
            print("[plugins] disabled:", exc)
            self.db = None
            self.dock = None

    def _add_toggle_to_view_menu(self) -> None:
        if not self.dock:
            return
        menu_view: Optional[QMenu] = self.window.findChild(QMenu, "menuView")
        if not menu_view:
            return
        action = QAction("Plugins", self.window)
        action.setCheckable(True)
        action.setChecked(self.dock.isVisible())
        action.toggled.connect(self.dock.setVisible)
        self.dock.visibilityChanged.connect(action.setChecked)
        menu_view.addAction(action)
        self.action_toggle = action

    def _ensure_plugins_menu(self) -> None:
        menubar = getattr(self.window, "menuBar", lambda: None)()
        if not isinstance(menubar, QMenuBar) or not PluginManagerDialog:
            return
        menu: Optional[QMenu] = None
        for action in menubar.actions():
            if action.menu() and action.text().replace("&", "").lower() == "plugins":
                menu = action.menu()
                break
        if menu is None:
            menu = QMenu("&Plugins", self.window)
            menubar.addMenu(menu)

        self.action_manager = QAction("Plugin Manager.", self.window)
        self.action_manager.triggered.connect(self._open_plugin_manager)
        menu.addAction(self.action_manager)

        self.action_add_folder = QAction("Add Plugin Folder.", self.window)
        self.action_add_folder.triggered.connect(self._add_plugin_folder)
        menu.addAction(self.action_add_folder)

    # -------------------------- menu actions --------------------------

    def _open_plugin_manager(self) -> None:
        if not PluginManagerDialog or not PluginDB or not self.db:
            QMessageBox.information(
                self.window,
                "Plugins",
                "Plugin features are unavailable in this build.",
            )
            return
        try:
            vm = PluginsViewModel(self.db)  # type: ignore[arg-type]
            dlg = PluginManagerDialog(vm, self.settings, self.window)
            code = dlg.exec()
            accepted = getattr(QDialog, "Accepted", None)
            if accepted is None:
                accepted = getattr(QDialog.DialogCode, "Accepted", 1)
            if code == accepted:
                text = getattr(dlg.ui.dirsEdit, "text", lambda: "")()
                dirs = [
                    p.strip()
                    for p in str(text).replace(";", ":").split(":")
                    if p.strip()
                ]
                if not dirs:
                    dirs = ["./plugins"]
                self.settings.plugin_dirs = dirs
                try:
                    self.settings.save()
                except Exception:
                    pass
                try:
                    self.db.set_setting("plugin_dirs", ":".join(dirs))  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._plugins_discover(dirs)
                try:
                    load_active_plugins(self.db)  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception as exc:
            QMessageBox.critical(self.window, "Plugin Manager", str(exc))

    def _add_plugin_folder(self) -> None:
        if not PluginDB or not self.db:
            QMessageBox.information(
                self.window,
                "Plugins",
                "Plugin features are unavailable in this build.",
            )
            return
        path = QFileDialog.getExistingDirectory(
            self.window, "Select Plugin Folder", str(Path.cwd())
        )
        if not path:
            return
        dirs = list(self.settings.plugin_dirs or [])
        if path not in dirs:
            dirs.append(path)
        self.settings.plugin_dirs = dirs
        try:
            self.settings.save()
        except Exception:
            pass
        try:
            self.db.set_setting("plugin_dirs", ":".join(dirs))  # type: ignore[attr-defined]
        except Exception:
            pass
        self._plugins_discover(dirs)
        try:
            load_active_plugins(self.db)  # type: ignore[arg-type]
        except Exception:
            pass

    # -------------------------- discovery --------------------------

    def _plugins_discover(self, dirs: Sequence[str | Path]) -> None:
        if not PluginDB or not self.db:
            return
        try:
            discover_into_db(self.db, [Path(d) for d in dirs])  # type: ignore[arg-type]
        except Exception as exc:
            print("[plugins] discover error:", exc)
