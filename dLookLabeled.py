# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dLookLabeled.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_dLookLabeled(object):
    def setupUi(self, dLookLabeled):
        dLookLabeled.setObjectName("dLookLabeled")
        dLookLabeled.resize(800, 660)
        self.centralwidget = QtWidgets.QWidget(dLookLabeled)
        self.centralwidget.setObjectName("centralwidget")
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setGeometry(QtCore.QRect(20, 20, 500, 500))
        self.label_img.setObjectName("label_img")
        self.list_choose = QtWidgets.QListWidget(self.centralwidget)
        self.list_choose.setGeometry(QtCore.QRect(530, 20, 251, 521))
        self.list_choose.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list_choose.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerItem)
        self.list_choose.setObjectName("list_choose")
        self.bLookOrigin = QtWidgets.QCommandLinkButton(self.centralwidget)
        self.bLookOrigin.setGeometry(QtCore.QRect(530, 550, 222, 48))
        self.bLookOrigin.setObjectName("bLookOrigin")
        dLookLabeled.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(dLookLabeled)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        dLookLabeled.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(dLookLabeled)
        self.statusbar.setObjectName("statusbar")
        dLookLabeled.setStatusBar(self.statusbar)

        self.retranslateUi(dLookLabeled)
        QtCore.QMetaObject.connectSlotsByName(dLookLabeled)

    def retranslateUi(self, dLookLabeled):
        _translate = QtCore.QCoreApplication.translate
        dLookLabeled.setWindowTitle(_translate("dLookLabeled", "查看已标注数据"))
        self.label_img.setText(_translate("dLookLabeled", "您还没有标注数据"))
        self.bLookOrigin.setText(_translate("dLookLabeled", "查看原图"))

