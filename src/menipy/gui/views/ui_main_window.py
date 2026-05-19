# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window_split.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
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
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        self.actionOpenImage = QAction(MainWindow)
        self.actionOpenImage.setObjectName("actionOpenImage")
        self.actionPreview = QAction(MainWindow)
        self.actionPreview.setObjectName("actionPreview")
        self.actionExportCsv = QAction(MainWindow)
        self.actionExportCsv.setObjectName("actionExportCsv")
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
        self.actionConfigOverlay = QAction(MainWindow)
        self.actionConfigOverlay.setObjectName("actionConfigOverlay")
        self.actionConfigMarkers = QAction(MainWindow)
        self.actionConfigMarkers.setObjectName("actionConfigMarkers")
        self.actionConfigPipeline = QAction(MainWindow)
        self.actionConfigPipeline.setObjectName("actionConfigPipeline")
        self.actionConfigPreprocessing = QAction(MainWindow)
        self.actionConfigPreprocessing.setObjectName("actionConfigPreprocessing")
        self.actionConfigEdgeDetection = QAction(MainWindow)
        self.actionConfigEdgeDetection.setObjectName("actionConfigEdgeDetection")
        self.actionConfigGeometry = QAction(MainWindow)
        self.actionConfigGeometry.setObjectName("actionConfigGeometry")
        self.actionConfigPhysics = QAction(MainWindow)
        self.actionConfigPhysics.setObjectName("actionConfigPhysics")
        self.actionConfigAcquisition = QAction(MainWindow)
        self.actionConfigAcquisition.setObjectName("actionConfigAcquisition")
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setNativeMenuBar(False)
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuConfig = QMenu(self.menubar)
        self.menuConfig.setObjectName("menuConfig")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuRun = QMenu(self.menubar)
        self.menuRun.setObjectName("menuRun")
        self.menuPlugins = QMenu(self.menubar)
        self.menuPlugins.setObjectName("menuPlugins")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName("centralLayout")
        self.workflowBar = QWidget(self.centralwidget)
        self.workflowBar.setObjectName("workflowBar")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.workflowBar.sizePolicy().hasHeightForWidth())
        self.workflowBar.setSizePolicy(sizePolicy)
        self.workflowBar.setMaximumSize(QSize(16777215, 44))
        self.workflowBarLayout = QHBoxLayout(self.workflowBar)
        self.workflowBarLayout.setSpacing(8)
        self.workflowBarLayout.setObjectName("workflowBarLayout")
        self.workflowBarLayout.setContentsMargins(8, 8, 8, 6)
        self.actionOpenImageBtn = QToolButton(self.workflowBar)
        self.actionOpenImageBtn.setObjectName("actionOpenImageBtn")
        self.actionOpenImageBtn.setAutoRaise(True)

        self.workflowBarLayout.addWidget(self.actionOpenImageBtn)

        self.workflowAutoCalibrateBtn = QToolButton(self.workflowBar)
        self.workflowAutoCalibrateBtn.setObjectName("workflowAutoCalibrateBtn")
        self.workflowAutoCalibrateBtn.setAutoRaise(True)

        self.workflowBarLayout.addWidget(self.workflowAutoCalibrateBtn)

        self.actionRunBtn = QToolButton(self.workflowBar)
        self.actionRunBtn.setObjectName("actionRunBtn")
        self.actionRunBtn.setAutoRaise(True)

        self.workflowBarLayout.addWidget(self.actionRunBtn)

        self.actionExportCsvBtn = QToolButton(self.workflowBar)
        self.actionExportCsvBtn.setObjectName("actionExportCsvBtn")
        self.actionExportCsvBtn.setAutoRaise(True)

        self.workflowBarLayout.addWidget(self.actionExportCsvBtn)

        self.workflowAdvancedBtn = QToolButton(self.workflowBar)
        self.workflowAdvancedBtn.setObjectName("workflowAdvancedBtn")
        self.workflowAdvancedBtn.setCheckable(True)
        self.workflowAdvancedBtn.setAutoRaise(True)

        self.workflowBarLayout.addWidget(self.workflowAdvancedBtn)

        self.workflowSpacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.workflowBarLayout.addItem(self.workflowSpacer)

        self.centralLayout.addWidget(self.workflowBar)

        self.rootSplitter = QSplitter(self.centralwidget)
        self.rootSplitter.setObjectName("rootSplitter")
        self.rootSplitter.setOrientation(Qt.Horizontal)
        self.setupHost = QWidget(self.rootSplitter)
        self.setupHost.setObjectName("setupHost")
        self.setupHostLayout = QVBoxLayout(self.setupHost)
        self.setupHostLayout.setObjectName("setupHostLayout")
        self.setupHostLayout.setContentsMargins(0, 0, 0, 0)
        self.rootSplitter.addWidget(self.setupHost)
        self.workbenchHost = QWidget(self.rootSplitter)
        self.workbenchHost.setObjectName("workbenchHost")
        self.workbenchHostLayout = QVBoxLayout(self.workbenchHost)
        self.workbenchHostLayout.setObjectName("workbenchHostLayout")
        self.workbenchHostLayout.setContentsMargins(0, 0, 0, 0)
        self.workbenchSplitter = QSplitter(self.workbenchHost)
        self.workbenchSplitter.setObjectName("workbenchSplitter")
        self.workbenchSplitter.setOrientation(Qt.Vertical)
        self.previewHost = QWidget(self.workbenchSplitter)
        self.previewHost.setObjectName("previewHost")
        self.previewHostLayout = QVBoxLayout(self.previewHost)
        self.previewHostLayout.setObjectName("previewHostLayout")
        self.previewHostLayout.setContentsMargins(0, 0, 0, 0)
        self.workbenchSplitter.addWidget(self.previewHost)
        self.inspectTabs = QTabWidget(self.workbenchSplitter)
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
        self.workbenchSplitter.addWidget(self.inspectTabs)

        self.workbenchHostLayout.addWidget(self.workbenchSplitter)

        self.rootSplitter.addWidget(self.workbenchHost)

        self.centralLayout.addWidget(self.rootSplitter)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuConfig.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menuPlugins.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionOpenImage)
        self.menuFile.addAction(self.actionOpenCamera)
        self.menuFile.addAction(self.actionExportCsv)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuConfig.addAction(self.actionConfigOverlay)
        self.menuConfig.addAction(self.actionConfigMarkers)
        self.menuConfig.addSeparator()
        self.menuConfig.addAction(self.actionConfigPipeline)
        self.menuConfig.addSeparator()
        self.menuConfig.addAction(self.actionConfigPreprocessing)
        self.menuConfig.addAction(self.actionConfigEdgeDetection)
        self.menuConfig.addAction(self.actionConfigGeometry)
        self.menuConfig.addAction(self.actionConfigPhysics)
        self.menuConfig.addAction(self.actionConfigAcquisition)
        self.menuView.addAction(self.actionOverlay)
        self.menuRun.addAction(self.actionPreview)
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
        self.actionPreview.setText(
            QCoreApplication.translate("MainWindow", "&Preview", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionPreview.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+P", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionExportCsv.setText(
            QCoreApplication.translate("MainWindow", "Export &CSV\u2026", None)
        )
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
        self.actionConfigOverlay.setText(
            QCoreApplication.translate("MainWindow", "Overlay Appearance\u2026", None)
        )
        # if QT_CONFIG(shortcut)
        self.actionConfigOverlay.setShortcut(
            QCoreApplication.translate("MainWindow", "Ctrl+Shift+O", None)
        )
        # endif // QT_CONFIG(shortcut)
        self.actionConfigMarkers.setText(
            QCoreApplication.translate("MainWindow", "Marker Display\u2026", None)
        )
        self.actionConfigPipeline.setText(
            QCoreApplication.translate("MainWindow", "Pipeline Settings\u2026", None)
        )
        self.actionConfigPreprocessing.setText(
            QCoreApplication.translate("MainWindow", "Preprocessing\u2026", None)
        )
        self.actionConfigEdgeDetection.setText(
            QCoreApplication.translate("MainWindow", "Edge Detection\u2026", None)
        )
        self.actionConfigGeometry.setText(
            QCoreApplication.translate("MainWindow", "Geometry\u2026", None)
        )
        self.actionConfigPhysics.setText(
            QCoreApplication.translate("MainWindow", "Physics\u2026", None)
        )
        self.actionConfigAcquisition.setText(
            QCoreApplication.translate("MainWindow", "Acquisition\u2026", None)
        )
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", "&File", None))
        self.menuConfig.setTitle(
            QCoreApplication.translate("MainWindow", "&Config", None)
        )
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", "&View", None))
        self.menuRun.setTitle(QCoreApplication.translate("MainWindow", "&Run", None))
        self.menuPlugins.setTitle(
            QCoreApplication.translate("MainWindow", "&Plugins", None)
        )
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", "&Help", None))
        self.actionOpenImageBtn.setText(
            QCoreApplication.translate("MainWindow", "Open Image", None)
        )
        self.workflowAutoCalibrateBtn.setText(
            QCoreApplication.translate("MainWindow", "Auto-Calibrate", None)
        )
        self.actionRunBtn.setText(
            QCoreApplication.translate("MainWindow", "Run Analysis", None)
        )
        self.actionExportCsvBtn.setText(
            QCoreApplication.translate("MainWindow", "Export CSV", None)
        )
        self.workflowAdvancedBtn.setText(
            QCoreApplication.translate("MainWindow", "Advanced +", None)
        )
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
