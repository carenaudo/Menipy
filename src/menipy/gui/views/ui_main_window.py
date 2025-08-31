# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QMenu,
    QMenuBar, QSizePolicy, QSpacerItem, QSplitter,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(666, 463)
        self.actionGota_pendiente = QAction(MainWindow)
        self.actionGota_pendiente.setObjectName(u"actionGota_pendiente")
        self.actionGota_s_sil = QAction(MainWindow)
        self.actionGota_s_sil.setObjectName(u"actionGota_s_sil")
        self.actionGota_oscilante = QAction(MainWindow)
        self.actionGota_oscilante.setObjectName(u"actionGota_oscilante")
        self.actionBurbuja_captiva = QAction(MainWindow)
        self.actionBurbuja_captiva.setObjectName(u"actionBurbuja_captiva")
        self.actionAscenso_capilar = QAction(MainWindow)
        self.actionAscenso_capilar.setObjectName(u"actionAscenso_capilar")
        self.action_plugins = QAction(MainWindow)
        self.action_plugins.setObjectName(u"action_plugins")
        self.action_plugins.setCheckable(True)
        self.action_plugins.setChecked(False)
        self.action_preview = QAction(MainWindow)
        self.action_preview.setObjectName(u"action_preview")
        self.action_preview.setCheckable(True)
        self.action_preview.setChecked(True)
        self.action_case = QAction(MainWindow)
        self.action_case.setObjectName(u"action_case")
        self.action_case.setCheckable(True)
        self.action_case.setChecked(True)
        self.action_results = QAction(MainWindow)
        self.action_results.setObjectName(u"action_results")
        self.action_results.setCheckable(True)
        self.action_results.setChecked(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName(u"centralLayout")
        self.topBar = QHBoxLayout()
        self.topBar.setObjectName(u"topBar")
        self.spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.topBar.addItem(self.spacer)


        self.centralLayout.addLayout(self.topBar)

        self.splitterH = QSplitter(self.centralwidget)
        self.splitterH.setObjectName(u"splitterH")
        self.splitterH.setOrientation(Qt.Orientation.Horizontal)
        self.splitterH.setChildrenCollapsible(False)
        self.runHost = QWidget(self.splitterH)
        self.runHost.setObjectName(u"runHost")
        self.runHostLayout = QVBoxLayout(self.runHost)
        self.runHostLayout.setObjectName(u"runHostLayout")
        self.runHostLayout.setContentsMargins(0, 0, 0, 0)
        self.splitterH.addWidget(self.runHost)
        self.splitterV = QSplitter(self.splitterH)
        self.splitterV.setObjectName(u"splitterV")
        self.splitterV.setOrientation(Qt.Orientation.Vertical)
        self.splitterV.setChildrenCollapsible(False)
        self.overlayHost = QWidget(self.splitterV)
        self.overlayHost.setObjectName(u"overlayHost")
        self.overlayHostLayout = QVBoxLayout(self.overlayHost)
        self.overlayHostLayout.setObjectName(u"overlayHostLayout")
        self.overlayHostLayout.setContentsMargins(0, 0, 0, 0)
        self.splitterV.addWidget(self.overlayHost)
        self.resultsHost = QWidget(self.splitterV)
        self.resultsHost.setObjectName(u"resultsHost")
        self.resultsHostLayout = QVBoxLayout(self.resultsHost)
        self.resultsHostLayout.setObjectName(u"resultsHostLayout")
        self.resultsHostLayout.setContentsMargins(0, 0, 0, 0)
        self.splitterV.addWidget(self.resultsHost)
        self.splitterH.addWidget(self.splitterV)

        self.centralLayout.addWidget(self.splitterH)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 666, 33))
        self.menuArchivo = QMenu(self.menuBar)
        self.menuArchivo.setObjectName(u"menuArchivo")
        self.menuEtapa = QMenu(self.menuBar)
        self.menuEtapa.setObjectName(u"menuEtapa")
        self.menuEtapa_2 = QMenu(self.menuBar)
        self.menuEtapa_2.setObjectName(u"menuEtapa_2")
        self.menuResultados = QMenu(self.menuBar)
        self.menuResultados.setObjectName(u"menuResultados")
        self.menuPlugins = QMenu(self.menuBar)
        self.menuPlugins.setObjectName(u"menuPlugins")
        self.menuAcerca_de = QMenu(self.menuBar)
        self.menuAcerca_de.setObjectName(u"menuAcerca_de")
        self.menuVistas = QMenu(self.menuBar)
        self.menuVistas.setObjectName(u"menuVistas")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuArchivo.menuAction())
        self.menuBar.addAction(self.menuEtapa.menuAction())
        self.menuBar.addAction(self.menuEtapa_2.menuAction())
        self.menuBar.addAction(self.menuResultados.menuAction())
        self.menuBar.addAction(self.menuPlugins.menuAction())
        self.menuBar.addAction(self.menuVistas.menuAction())
        self.menuBar.addAction(self.menuAcerca_de.menuAction())
        self.menuEtapa.addAction(self.actionGota_pendiente)
        self.menuEtapa.addAction(self.actionGota_s_sil)
        self.menuEtapa.addAction(self.actionGota_oscilante)
        self.menuEtapa.addAction(self.actionBurbuja_captiva)
        self.menuEtapa.addAction(self.actionAscenso_capilar)
        self.menuVistas.addAction(self.action_plugins)
        self.menuVistas.addAction(self.action_preview)
        self.menuVistas.addAction(self.action_case)
        self.menuVistas.addAction(self.action_results)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Menipy GUI", None))
        self.actionGota_pendiente.setText(QCoreApplication.translate("MainWindow", u"Gota pendiente", None))
        self.actionGota_s_sil.setText(QCoreApplication.translate("MainWindow", u"Gota s\u00e9sil", None))
        self.actionGota_oscilante.setText(QCoreApplication.translate("MainWindow", u"Gota oscilante", None))
        self.actionBurbuja_captiva.setText(QCoreApplication.translate("MainWindow", u"Burbuja captiva", None))
        self.actionAscenso_capilar.setText(QCoreApplication.translate("MainWindow", u"Ascenso capilar", None))
        self.action_plugins.setText(QCoreApplication.translate("MainWindow", u"Plugins", None))
        self.action_preview.setText(QCoreApplication.translate("MainWindow", u"Imagen", None))
        self.action_preview.setIconText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.action_case.setText(QCoreApplication.translate("MainWindow", u"Configuraci\u00f3n", None))
        self.action_results.setText(QCoreApplication.translate("MainWindow", u"Reporte", None))
        self.menuArchivo.setTitle(QCoreApplication.translate("MainWindow", u"Archivo", None))
        self.menuEtapa.setTitle(QCoreApplication.translate("MainWindow", u"Ensayo", None))
        self.menuEtapa_2.setTitle(QCoreApplication.translate("MainWindow", u"Etapa", None))
        self.menuResultados.setTitle(QCoreApplication.translate("MainWindow", u"Resultados", None))
        self.menuPlugins.setTitle(QCoreApplication.translate("MainWindow", u"Plugins", None))
        self.menuAcerca_de.setTitle(QCoreApplication.translate("MainWindow", u"Acerca de", None))
        self.menuVistas.setTitle(QCoreApplication.translate("MainWindow", u"Vistas", None))
    # retranslateUi

