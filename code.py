import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from photo_view import Ui_Photo_view
from PIL import Image
import cv2
import my_yolo
import numpy as np
class show_photo(QMainWindow):
	def __init__(self):
		super(show_photo,self).__init__()
		self.ui =Ui_Photo_view()
		self.ui.setupUi(self)
		self.view = QGraphicsView(self)
		self.view.setGeometry(0,0,1280,720)
		
	def select_button_clicked(self):
		file_name = QFileDialog.getOpenFileName(self, "Open File", "./", "jpg (*.jpg)")
		image_path = file_name[0]
		if (file_name[0] == ""):
			QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
            
            
		file_name2 = QFileDialog.getOpenFileName(self, "Open File", "./", "jpg (*.jpg)")
		image_path2 = file_name2[0]
		if (file_name2[0] == ""):
			QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
            
		print(image_path,image_path2)
		img=my_yolo.yolo_my_img(image_path,image_path2)      
		img = np.asanyarray(img,dtype=np.uint8)
# 		img = cv2.imread(image_path)  # 读取图像
# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
# 		m=0.9
		x = img.shape[1]  # 获取图像大小
		y = img.shape[0]
# 		x=int(x*m);y=int(y*m)
# # 		
# 		img = cv2.resize(img,(x,y))
        
        
# # 		self.zoomscale = 0.3  # 图片放缩尺度
		frame = QImage(img, x, y, QImage.Format_RGB888)
		pix = QPixmap.fromImage(frame)
		self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
		self.scene = QGraphicsScene()  # 创建场景
		self.scene.addItem(self.item)
		self.view.setScene(self.scene)
		self.view.show()
		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	mainWindow = show_photo()
	mainWindow.showMaximized()
	sys.exit(app.exec_())