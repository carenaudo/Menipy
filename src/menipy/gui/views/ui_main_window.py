"""
UI definition for main window (likely auto-generated).

Note: This file may be auto-generated. Avoid manual modifications.
"""

# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window_split.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        self.actionOpenImage = QAction(MainWindow)
        self.actionOpenImage.setObjectName("actionOpenImage")
        self.actionOpenCamera = QAction(MainWindow)
        self.actionOpenCamera.setObjectName("actionOpenCamera")
        self.actionRunFull = QAction(MainWindow)
        self.actionRunFull.setObjectName("actionRunFull")
        self.actionRunSelected = QAction(MainWindow)
        self.actionRunSelected.setObjectName("actionRunSelected")
        self.actionStop = QAction(MainWindow)
        self.actionStop.setObjectName("actionStop")
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionOverlay = QAction(MainWindow)
        self.actionOverlay.setObjectName("actionOverlay")
        icon = QIcon()
        icon.addFile(":/icons/overlay.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionOverlay.setIcon(icon)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setNativeMenuBar(False)
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuRun = QMenu(self.menubar)
        self.menuRun.setObjectName("menuRun")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName("centralLayout")
        self.rootSplitter = QSplitter(self.centralwidget)
        self.rootSplitter.setObjectName("rootSplitter")
        self.rootSplitter.setOrientation(Qt.Horizontal)
        self.setupHost = QWidget(self.rootSplitter)
        self.setupHost.setObjectName("setupHost")
        self.setupHostLayout = QVBoxLayout(self.setupHost)
        self.setupHostLayout.setObjectName("setupHostLayout")
        self.setupHostLayout.setContentsMargins(0, 0, 0, 0)
        self.rootSplitter.addWidget(self.setupHost)
        self.previewHost = QWidget(self.rootSplitter)
        self.previewHost.setObjectName("previewHost")
        self.previewHostLayout = QVBoxLayout(self.previewHost)
        self.previewHostLayout.setObjectName("previewHostLayout")
        self.previewHostLayout.setContentsMargins(0, 0, 0, 0)
        self.rootSplitter.addWidget(self.previewHost)
        self.inspectTabs = QTabWidget(self.rootSplitter)
        self.inspectTabs.setObjectName("inspectTabs")
        self.inspectTabs.setTabPosition(QTabWidget.North)
        self.resultsTab = QWidget()
        self.resultsTab.setObjectName("resultsTab")
        self.resultsHostLayout = QVBoxLayout(self.resultsTab)
        self.resultsHostLayout.setObjectName("resultsHostLayout")
        self.inspectTabs.addTab(self.resultsTab, "")
        self.residualsTab = QWidget()
        self.residualsTab.setObjectName("residualsTab")
        self.residualsHostLayout = QVBoxLayout(self.residualsTab)
        self.residualsHostLayout.setObjectName("residualsHostLayout")
        self.inspectTabs.addTab(self.residualsTab, "")
        self.timingsTab = QWidget()
        self.timingsTab.setObjectName("timingsTab")
        self.timingsHostLayout = QVBoxLayout(self.timingsTab)
        self.timingsHostLayout.setObjectName("timingsHostLayout")
        self.inspectTabs.addTab(self.timingsTab, "")
        self.logTab = QWidget()
        self.logTab.setObjectName("logTab")
        self.logHostLayout = QVBoxLayout(self.logTab)
        self.logHostLayout.setObjectName("logHostLayout")
        self.inspectTabs.addTab(self.logTab, "")
        self.rootSplitter.addWidget(self.inspectTabs)

        self.centralLayout.addWidget(self.rootSplitter)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionOpenImage)
        self.menuFile.addAction(self.actionOpenCamera)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuView.addAction(self.actionOverlay)
        self.menuRun.addAction(self.actionRunFull)
        self.menuRun.addAction(self.actionRunSelected)
        self.menuRun.addSeparator()
        self.menuRun.addAction(self.actionStop)
        self.menuHelp.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Menipy ADSA", None)
        )
        self.actionOpenImage.setText(
            QCoreApplication.translate("MainWindow", "Open &Image\u2026", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionOpenImage.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+O", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionOpenCamera.setText(
            QCoreApplication.translate("MainWindow", "Open &Camera", None)
        )
        self.actionRunFull.setText(
            QCoreApplication.translate("MainWindow", "Run &Full", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionRunFull.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+R", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionRunSelected.setText(
            QCoreApplication.translate("MainWindow", "Run &Selected", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionRunSelected.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+Shift+R", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionStop.setText(QCoreApplication.translate("MainWindow", "&Stop", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", "&Quit", None))
        # if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+Q", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionAbout.setText(
            QCoreApplication.translate("MainWindow", "&About", None)
        )
        self.actionOverlay.setText(
            QCoreApplication.translate("MainWindow", "&Overlay\u2026", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionOverlay.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+Shift+O", None)
        )
        # endif // QT_CONFIG(shortcut)
        # if QT_CONFIG(tooltip)
        self.actionOverlay.setToolTip(
            QCoreApplication.translate(
                "MainWindow",
                "Open overlay configuration and preview overlay styling",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.actionOverlay.setStatusTip(
            QCoreApplication.translate(
                "MainWindow", "Configure overlay appearance", None
            )
        )
        # endif // QT_CONFIG(statustip)
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", "&File", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", "&View", None))
        self.menuRun.setTitle(QCoreApplication.translate("MainWindow", "&Run", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", "&Help", None))
        self.inspectTabs.setTabText(
            self.inspectTabs.indexOf(self.resultsTab),
            QCoreApplication.translate("MainWindow", "Results", None),
        )
        self.inspectTabs.setTabText(
            self.inspectTabs.indexOf(self.residualsTab),
            QCoreApplication.translate("MainWindow", "Residuals", None),
        )
        self.inspectTabs.setTabText(
            self.inspectTabs.indexOf(self.timingsTab),
            QCoreApplication.translate("MainWindow", "Timings", None),
        )
        self.inspectTabs.setTabText(
            self.inspectTabs.indexOf(self.logTab),
            QCoreApplication.translate("MainWindow", "Log", None),
        )

    # retranslateUi
