# -*- coding: utf-8 -*-
"""Ui Main Window.

Module implementation."""



################################################################################
## Form generated from reading UI file 'main_window_split.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QMenuBar, QSizePolicy,
    QSpacerItem, QSplitter, QStatusBar, QTabWidget,
    QToolButton, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        self.actionOpenImage = QAction(MainWindow)
        self.actionOpenImage.setObjectName(u"actionOpenImage")
        self.actionPreview = QAction(MainWindow)
        self.actionPreview.setObjectName(u"actionPreview")
        self.actionExportCsv = QAction(MainWindow)
        self.actionExportCsv.setObjectName(u"actionExportCsv")
        self.actionOpenCamera = QAction(MainWindow)
        self.actionOpenCamera.setObjectName(u"actionOpenCamera")
        self.actionRunFull = QAction(MainWindow)
        self.actionRunFull.setObjectName(u"actionRunFull")
        self.actionRunSelected = QAction(MainWindow)
        self.actionRunSelected.setObjectName(u"actionRunSelected")
        self.actionStop = QAction(MainWindow)
        self.actionStop.setObjectName(u"actionStop")
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.actionOverlay = QAction(MainWindow)
        self.actionOverlay.setObjectName(u"actionOverlay")
        icon = QIcon()
        icon.addFile(u":/icons/overlay.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.actionOverlay.setIcon(icon)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setNativeMenuBar(False)
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuRun = QMenu(self.menubar)
        self.menuRun.setObjectName(u"menuRun")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName(u"centralLayout")
        self.layoutBar = QWidget(self.centralwidget)
        self.layoutBar.setObjectName(u"layoutBar")
        self.layoutBarLayout = QHBoxLayout(self.layoutBar)
        self.layoutBarLayout.setSpacing(8)
        self.layoutBarLayout.setObjectName(u"layoutBarLayout")
        self.layoutBarLayout.setContentsMargins(8, 8, 8, 0)
        self.layoutLabel = QLabel(self.layoutBar)
        self.layoutLabel.setObjectName(u"layoutLabel")

        self.layoutBarLayout.addWidget(self.layoutLabel)

        self.layoutAnalysisBtn = QToolButton(self.layoutBar)
        self.layoutAnalysisBtn.setObjectName(u"layoutAnalysisBtn")
        self.layoutAnalysisBtn.setCheckable(True)
        self.layoutAnalysisBtn.setAutoExclusive(True)

        self.layoutBarLayout.addWidget(self.layoutAnalysisBtn)

        self.layoutSetupBtn = QToolButton(self.layoutBar)
        self.layoutSetupBtn.setObjectName(u"layoutSetupBtn")
        self.layoutSetupBtn.setCheckable(True)
        self.layoutSetupBtn.setAutoExclusive(True)

        self.layoutBarLayout.addWidget(self.layoutSetupBtn)

        self.layoutReviewBtn = QToolButton(self.layoutBar)
        self.layoutReviewBtn.setObjectName(u"layoutReviewBtn")
        self.layoutReviewBtn.setCheckable(True)
        self.layoutReviewBtn.setChecked(True)
        self.layoutReviewBtn.setAutoExclusive(True)

        self.layoutBarLayout.addWidget(self.layoutReviewBtn)

        self.layoutSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.layoutBarLayout.addItem(self.layoutSpacer)

        self.toggleSetupBtn = QToolButton(self.layoutBar)
        self.toggleSetupBtn.setObjectName(u"toggleSetupBtn")
        self.toggleSetupBtn.setCheckable(True)
        self.toggleSetupBtn.setChecked(True)

        self.layoutBarLayout.addWidget(self.toggleSetupBtn)

        self.toggleInspectBtn = QToolButton(self.layoutBar)
        self.toggleInspectBtn.setObjectName(u"toggleInspectBtn")
        self.toggleInspectBtn.setCheckable(True)
        self.toggleInspectBtn.setChecked(True)

        self.layoutBarLayout.addWidget(self.toggleInspectBtn)


        self.centralLayout.addWidget(self.layoutBar)

        self.rootSplitter = QSplitter(self.centralwidget)
        self.rootSplitter.setObjectName(u"rootSplitter")
        self.rootSplitter.setOrientation(Qt.Horizontal)
        self.setupHost = QWidget(self.rootSplitter)
        self.setupHost.setObjectName(u"setupHost")
        self.setupHostLayout = QVBoxLayout(self.setupHost)
        self.setupHostLayout.setObjectName(u"setupHostLayout")
        self.setupHostLayout.setContentsMargins(0, 0, 0, 0)
        self.rootSplitter.addWidget(self.setupHost)
        self.previewHost = QWidget(self.rootSplitter)
        self.previewHost.setObjectName(u"previewHost")
        self.previewHostLayout = QVBoxLayout(self.previewHost)
        self.previewHostLayout.setObjectName(u"previewHostLayout")
        self.previewHostLayout.setContentsMargins(0, 0, 0, 0)
        self.rootSplitter.addWidget(self.previewHost)
        self.inspectTabs = QTabWidget(self.rootSplitter)
        self.inspectTabs.setObjectName(u"inspectTabs")
        self.inspectTabs.setTabPosition(QTabWidget.North)
        self.resultsTab = QWidget()
        self.resultsTab.setObjectName(u"resultsTab")
        self.resultsHostLayout = QVBoxLayout(self.resultsTab)
        self.resultsHostLayout.setObjectName(u"resultsHostLayout")
        self.inspectTabs.addTab(self.resultsTab, "")
        self.residualsTab = QWidget()
        self.residualsTab.setObjectName(u"residualsTab")
        self.residualsHostLayout = QVBoxLayout(self.residualsTab)
        self.residualsHostLayout.setObjectName(u"residualsHostLayout")
        self.inspectTabs.addTab(self.residualsTab, "")
        self.timingsTab = QWidget()
        self.timingsTab.setObjectName(u"timingsTab")
        self.timingsHostLayout = QVBoxLayout(self.timingsTab)
        self.timingsHostLayout.setObjectName(u"timingsHostLayout")
        self.inspectTabs.addTab(self.timingsTab, "")
        self.logTab = QWidget()
        self.logTab.setObjectName(u"logTab")
        self.logHostLayout = QVBoxLayout(self.logTab)
        self.logHostLayout.setObjectName(u"logHostLayout")
        self.inspectTabs.addTab(self.logTab, "")
        self.rootSplitter.addWidget(self.inspectTabs)

        self.centralLayout.addWidget(self.rootSplitter)

        self.actionBar = QFrame(self.centralwidget)
        self.actionBar.setObjectName(u"actionBar")
        self.actionBar.setFrameShape(QFrame.StyledPanel)
        self.actionBar.setFrameShadow(QFrame.Plain)
        self.actionBarLayout = QHBoxLayout(self.actionBar)
        self.actionBarLayout.setSpacing(10)
        self.actionBarLayout.setObjectName(u"actionBarLayout")
        self.actionBarLayout.setContentsMargins(8, 6, 8, 8)
        self.sessionLabel = QLabel(self.actionBar)
        self.sessionLabel.setObjectName(u"sessionLabel")

        self.actionBarLayout.addWidget(self.sessionLabel)

        self.actionOpenImageBtn = QToolButton(self.actionBar)
        self.actionOpenImageBtn.setObjectName(u"actionOpenImageBtn")
        self.actionOpenImageBtn.setAutoRaise(True)

        self.actionBarLayout.addWidget(self.actionOpenImageBtn)

        self.sessionSeparator = QFrame(self.actionBar)
        self.sessionSeparator.setObjectName(u"sessionSeparator")
        self.sessionSeparator.setFrameShape(QFrame.VLine)
        self.sessionSeparator.setFrameShadow(QFrame.Sunken)

        self.actionBarLayout.addWidget(self.sessionSeparator)

        self.processingLabel = QLabel(self.actionBar)
        self.processingLabel.setObjectName(u"processingLabel")

        self.actionBarLayout.addWidget(self.processingLabel)

        self.actionPreviewBtn = QToolButton(self.actionBar)
        self.actionPreviewBtn.setObjectName(u"actionPreviewBtn")
        self.actionPreviewBtn.setAutoRaise(True)

        self.actionBarLayout.addWidget(self.actionPreviewBtn)

        self.actionRunBtn = QToolButton(self.actionBar)
        self.actionRunBtn.setObjectName(u"actionRunBtn")
        self.actionRunBtn.setAutoRaise(True)

        self.actionBarLayout.addWidget(self.actionRunBtn)

        self.actionStopBtn = QToolButton(self.actionBar)
        self.actionStopBtn.setObjectName(u"actionStopBtn")
        self.actionStopBtn.setAutoRaise(True)

        self.actionBarLayout.addWidget(self.actionStopBtn)

        self.processingSeparator = QFrame(self.actionBar)
        self.processingSeparator.setObjectName(u"processingSeparator")
        self.processingSeparator.setFrameShape(QFrame.VLine)
        self.processingSeparator.setFrameShadow(QFrame.Sunken)

        self.actionBarLayout.addWidget(self.processingSeparator)

        self.exportLabel = QLabel(self.actionBar)
        self.exportLabel.setObjectName(u"exportLabel")

        self.actionBarLayout.addWidget(self.exportLabel)

        self.actionExportCsvBtn = QToolButton(self.actionBar)
        self.actionExportCsvBtn.setObjectName(u"actionExportCsvBtn")
        self.actionExportCsvBtn.setAutoRaise(True)

        self.actionBarLayout.addWidget(self.actionExportCsvBtn)

        self.actionSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.actionBarLayout.addItem(self.actionSpacer)


        self.centralLayout.addWidget(self.actionBar)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionOpenImage)
        self.menuFile.addAction(self.actionOpenCamera)
        self.menuFile.addAction(self.actionExportCsv)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
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
        """retranslateUi.

        Parameters
        ----------
        MainWindow : type
        Description.
        """
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Menipy ADSA", None))
        self.actionOpenImage.setText(QCoreApplication.translate("MainWindow", u"Open &Image\u2026", None))
        #if QT_CONFIG(shortcut)
        self.actionOpenImage.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
        #endif // QT_CONFIG(shortcut)
        self.actionPreview.setText(QCoreApplication.translate("MainWindow", u"&Preview", None))
        #if QT_CONFIG(shortcut)
        self.actionPreview.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+P", None))
        #endif // QT_CONFIG(shortcut)
        self.actionExportCsv.setText(QCoreApplication.translate("MainWindow", u"Export &CSV\u2026", None))
        self.actionOpenCamera.setText(QCoreApplication.translate("MainWindow", u"Open &Camera", None))
        self.actionRunFull.setText(QCoreApplication.translate("MainWindow", u"Run &Full", None))
        #if QT_CONFIG(shortcut)
        self.actionRunFull.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+R", None))
        #endif // QT_CONFIG(shortcut)
        self.actionRunSelected.setText(QCoreApplication.translate("MainWindow", u"Run &Selected", None))
        #if QT_CONFIG(shortcut)
        self.actionRunSelected.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+R", None))
        #endif // QT_CONFIG(shortcut)
        self.actionStop.setText(QCoreApplication.translate("MainWindow", u"&Stop", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"&Quit", None))
        #if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Q", None))
        #endif // QT_CONFIG(shortcut)
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"&About", None))
        self.actionOverlay.setText(QCoreApplication.translate("MainWindow", u"&Overlay\u2026", None))
        #if QT_CONFIG(shortcut)
        self.actionOverlay.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+O", None))
        #endif // QT_CONFIG(shortcut)
        #if QT_CONFIG(tooltip)
        self.actionOverlay.setToolTip(QCoreApplication.translate("MainWindow", u"Open overlay configuration and preview overlay styling", None))
        #endif // QT_CONFIG(tooltip)
        #if QT_CONFIG(statustip)
        self.actionOverlay.setStatusTip(QCoreApplication.translate("MainWindow", u"Configure overlay appearance", None))
        #endif // QT_CONFIG(statustip)
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"&File", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", u"&View", None))
        self.menuRun.setTitle(QCoreApplication.translate("MainWindow", u"&Run", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"&Help", None))
        self.layoutLabel.setText(QCoreApplication.translate("MainWindow", u"Layout", None))
        self.layoutAnalysisBtn.setText(QCoreApplication.translate("MainWindow", u"Analysis", None))
        #if QT_CONFIG(tooltip)
        self.layoutAnalysisBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Focus on the image canvas", None))
        #endif // QT_CONFIG(tooltip)
        self.layoutSetupBtn.setText(QCoreApplication.translate("MainWindow", u"Setup", None))
        #if QT_CONFIG(tooltip)
        self.layoutSetupBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Show setup and preview panels", None))
        #endif // QT_CONFIG(tooltip)
        self.layoutReviewBtn.setText(QCoreApplication.translate("MainWindow", u"Review", None))
        #if QT_CONFIG(tooltip)
        self.layoutReviewBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Show setup, preview, and inspector panels", None))
        #endif // QT_CONFIG(tooltip)
        self.toggleSetupBtn.setText(QCoreApplication.translate("MainWindow", u"Setup Panel", None))
        #if QT_CONFIG(tooltip)
        self.toggleSetupBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Show or hide the setup panel", None))
        #endif // QT_CONFIG(tooltip)
        self.toggleInspectBtn.setText(QCoreApplication.translate("MainWindow", u"Inspector", None))
        #if QT_CONFIG(tooltip)
        self.toggleInspectBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Show or hide the results inspector", None))
        #endif // QT_CONFIG(tooltip)
        self.inspectTabs.setTabText(self.inspectTabs.indexOf(self.resultsTab), QCoreApplication.translate("MainWindow", u"Results", None))
        self.inspectTabs.setTabText(self.inspectTabs.indexOf(self.residualsTab), QCoreApplication.translate("MainWindow", u"Residuals", None))
        self.inspectTabs.setTabText(self.inspectTabs.indexOf(self.timingsTab), QCoreApplication.translate("MainWindow", u"Timings", None))
        self.inspectTabs.setTabText(self.inspectTabs.indexOf(self.logTab), QCoreApplication.translate("MainWindow", u"Log", None))
        self.sessionLabel.setText(QCoreApplication.translate("MainWindow", u"Session", None))
        self.actionOpenImageBtn.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.processingLabel.setText(QCoreApplication.translate("MainWindow", u"Processing", None))
        self.actionPreviewBtn.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.actionRunBtn.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.actionStopBtn.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.exportLabel.setText(QCoreApplication.translate("MainWindow", u"Export", None))
        self.actionExportCsvBtn.setText(QCoreApplication.translate("MainWindow", u"Export CSV", None))
        # retranslateUi

