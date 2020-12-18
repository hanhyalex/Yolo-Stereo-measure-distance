# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:48:46 2020

@author: 86198
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Photo_view(object):
    def setupUi(self, Photo_view):
        Photo_view.setObjectName("Photo_view")
        # Photo_view.resize(1280,720)
        
        self.centralwidget = QtWidgets.QWidget(Photo_view)
        self.centralwidget.setObjectName("centralwidget")
        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(1280, 150, 75, 23))
        self.select_button.setCheckable(True)
        self.select_button.setObjectName("select_button")
        self.pic_show = QtWidgets.QGraphicsView(self.centralwidget)
        self.pic_show.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        
        
        
        
        
        self.pic_show.setObjectName("pic_show")
        Photo_view.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(Photo_view)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 890, 23))
        # self.menubar.setObjectName("menubar")
        # Photo_view.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(Photo_view)
        # self.statusbar.setObjectName("statusbar")
        # Photo_view.setStatusBar(self.statusbar)

        self.retranslateUi(Photo_view)
        self.select_button.clicked.connect(Photo_view.select_button_clicked)
        QtCore.QMetaObject.connectSlotsByName(Photo_view)

    def retranslateUi(self, Photo_view):
        _translate = QtCore.QCoreApplication.translate
        Photo_view.setWindowTitle(_translate("Photo_view", "MainWindow"))
        self.select_button.setText(_translate("Photo_view", "选择文件"))