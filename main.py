#!/home/tqc/anaconda3/envs/tf/bin/python
#coding: utf-8
import json
import math
import os,platform
import shutil
import sys
import webbrowser
from queue import Queue
from select import select

import cv2
import numpy as np
import pylab as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *  # QMessageBox,QFileDialog,QWidget,QApplication,QMainWindow,QSizePolicy
from scipy.misc import imsave
from skimage import morphology

# import FCN_network
import utils
# 查看快捷键表
from dKeyTable import Ui_dKeyTable
# 查看已标注
from dLookLabeled import Ui_dLookLabeled
from labeler import *

keyboard_name = 'keyboard.json'

pluginsPath='PyQt5/Qt/plugins'
if os.path.exists(pluginsPath):#指定插件路径。源码运行时不会生效，打包后运行检测到路径，加载插件
    QApplication.addLibraryPath(pluginsPath)

global gWorkPath
# global gWin
global lookLabelWin
global LEN
global GX
global GY
LEN=656
# open_FCN=True
open_FCN=False
detectInputKeyInLinux=False
tool2name={'pen':'画笔','line':'直线','rubber':'橡皮擦','bucket':'油漆桶','select':'选择'}
red=(255,0,0)
blue=(0,0,255)

class My_dKeyTable(QMainWindow,Ui_dKeyTable):
    def __init__(self, parent=None):
        super(My_dKeyTable, self).__init__(parent)
        self.setupUi(self)
        global LEN
        self.table.setColumnWidth(3, LEN)
        hei=50
        self.table.setRowHeight(5, hei)
        self.table.setRowHeight(6, hei)
        self.table.setRowHeight(8, hei)
        self.table.setRowHeight(10, hei)
    def view(self):
        if not self.isVisible():
            self.show()

class My_dLookLabeled(QMainWindow,Ui_dLookLabeled):
    def __init__(self, parent=None):
        super(My_dLookLabeled, self).__init__(parent)
        self.setupUi(self)
        self.list_choose.itemClicked.connect(self.choose)
        self.bLookOrigin.clicked.connect(self.lookOrigin)
        self.list_choose.itemSelectionChanged.connect(self.choose2)
        self.img=[]
        self.data=[]
        self.plus=[]
        self.dataPath=''
        self.imgPath=''

    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def view(self):
        if not self.isVisible():
            self.show()
            self.setData()

    def setData(self):
        global gWorkPath
        if not gWorkPath :return
        self.list_choose.clear()
        self.dataPath=gWorkPath+'/标注数据'
        self.imgPath=gWorkPath+'/标注图片'
        if not os.path.exists(self.dataPath): return
        for (root, dirs, files) in os.walk(self.dataPath):
            if not files:
                # QMessageBox.warning(self,'警告','该文件夹下没有文件！')
                return
            for f in files:
                name=os.path.splitext(f)[0]
                self.list_choose.addItem(name)
            break

    def choose(self,item):
        if not item: return
        name=item.text()
        if not name: return
        # self.data=np.load(self.dataPath+'/'+name+'.npy')
        self.data=plt.imread(self.dataPath+'/'+name+'.jpg')
        self.img=plt.imread(self.imgPath+'/'+name+'.jpg')
        self.plus=self.imgPlus(self.img,self.data)
        self.label_img.setPixmap(self.img2pixmap(self.plus))
        self.initLookOrigin()

    def choose2(self):
        if len(self.list_choose.selectedItems())>0:
            self.choose(self.list_choose.selectedItems()[0])

    def imgPlus(self, a, b):
        ans = a.copy()
        x, y = b.shape[:2]
        for i in range(0, x):
            for j in range(0, y):
                if b[i, j] == 255:
                    ans[i, j, 0] = 255
                    ans[i, j, 1] = 0
                    ans[i, j, 2] = 0
        return ans

    def lookOrigin(self):
        strOrigin='查看原图[Alt]'
        strLabel='查看标注[Alt]'
        if self.bLookOrigin.text()==strOrigin:
            self.bLookOrigin.setText(strLabel)
            plt.imsave('tmp.png', self.img)
            self.label_img.setPixmap(QPixmap('tmp.png'))
        else:
            self.bLookOrigin.setText(strOrigin)
            plt.imsave('tmp.png', self.plus)
            self.label_img.setPixmap(QPixmap('tmp.png'))

    def keyPressEvent(self, event):
        key=""
        if event.modifiers() & Qt.AltModifier:
            self.lookOrigin()
            key = ' '
        if key:
            self.update()
        else:
            QWidget.keyPressEvent(self, event)

class MyMainWindow(QMainWindow,Ui_MainWindow):

    def buildConnect(self):
        self.cb_alpha.currentIndexChanged.connect(self.alphaChange)
        self.b_openImg.clicked.connect(self.openImg)
        self.b_up.clicked.connect(self.offsetUp)
        self.b_down.clicked.connect(self.offsetDown)
        self.b_left.clicked.connect(self.offsetLeft)
        self.b_right.clicked.connect(self.offsetRight)
        self.bChooseImg.clicked.connect(self.chooseImg)  # 选择图片
        self.toolButton.clicked.connect(self.chooseFold)  # 选择路径（工具按钮）
        self.bChooseFloder.clicked.connect(self.chooseFold)  # 选择路径
        self.bFinish.clicked.connect(self.nextImg)  # 完成标注
        self.bFlash.clicked.connect(self.flashImg)  # 刷新
        self.sB1.valueChanged.connect(self.usingFlash)  # 微调器改变
        self.sB1.valueChanged.connect(self.usingFlash)
        self.sSwell.valueChanged.connect(self.usingFlash)
        self.sConnect.valueChanged.connect(self.usingFlash)
        self.cDel.clicked.connect(self.usingFlash)
        self.cFill.clicked.connect(self.usingFlash)
        self.bUndo.clicked.connect(self.UnDo)
        self.bRedo.clicked.connect(self.ReDo)
        self.bHelp.clicked.connect(self.Help)
        self.bPre.clicked.connect(self.preImg)
        self.bAbandon.clicked.connect(self.abandonImg)
        self.bClear.clicked.connect(self.clearData)
        self.bTable.clicked.connect(self.lookTable)
        self.sPaint.valueChanged.connect(self.paintFlash)
        self.bPaint.clicked.connect(self.paintFlash)
        self.bAmend.clicked.connect(self.amendData)
        self.bSwell.clicked.connect(self.Swell)
        self.bErosion.clicked.connect(self.Erosion)
        self.b_release_select.clicked.connect(self.release_select)
    def initData(self):
        # self.preAlpha='30'
        self.select=None
        self.preDir = None
        self.prePt = None  # 前驱点
        self.toolMode = ''  # pen 或 rubber
        self.stateMessage = ''  # 状态栏信息
        self.workPath = ''  # 工作路径
        self.workFile = ''  # 当前工作文件
        self.img = None  # 原图片
        self.plus = None  # 标注图片
        self.data = None  # 标注数据
        self.plusList = []  # 标注图片列表，用于撤销
        self.dataList = []  # 标注数据列表，用于撤销
        self.undoPos = 0  # 还原点，用于撤销
        self.mouseIsPress = False
        self.rubber_is_used = False
        self.pressX=False
        self.tabWidget.setTabText(0,'边缘检测算法')
        self.tabWidget.setTabText(1,'FCN算法')
        #工具栏
        # tb=self.addToolBar('绘图') #添加图形按钮
        button = QAction(QIcon('res/pen.png'), 'pen', self)
        button.setShortcut('Ctrl+P')
        self.tb.addAction(button)
        button = QAction(QIcon('res/rubber.png'), 'rubber', self)
        button.setShortcut('Ctrl+R')
        self.tb.addAction(button)
        button = QAction(QIcon('res/line.png'), 'line', self)
        button.setShortcut('Ctrl+L')
        self.tb.addAction(button)
        button = QAction(QIcon('res/bucket.png'), 'bucket', self)
        button.setShortcut('Ctrl+B')
        self.tb.addAction(button)
        button = QAction(QIcon('res/select.png'), 'select', self)
        button.setShortcut('Ctrl+S')
        self.tb.addAction(button)
        #
        self.tb.setMovable(True)
        self.set_current_tool('pen')
        self.tb.actionTriggered[QAction].connect(self.tool_btn_pressed)
        # self.spinBox.hide()

    def set_current_tool(self,toolMode):
        self.toolMode=toolMode
        self.l_current_tool.setText(tool2name[self.toolMode])
        self.l_current_tool_image.setPixmap(QPixmap('res/'+self.toolMode+'.png'))
        self.prePt=None

    def __init__(self,parent=None):
        #加载模型
        # if open_FCN:
        #     self.FCN=FCN_network.FCN()
        #     self.FCN.init_net()
        #     print('模型加载完成')
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon("res/logo.png")) #设置图标
        self.setMouseTracking(True)
        self.initData()
        global GX
        global GY
        GX = self.lable_img.pos().x()
        GY = self.lable_img.pos().y()#+self.tb.rect().height()

        self.buildConnect()
        self.loadConfigure(
            (self.sB1,400),
            (self.sB2,200),
            (self.sSwell,2),
            (self.sConnect,100),
            (self.cFill,1),
            (self.cDel,1),
            (self.sPen,5),
            (self.sRubber,20),
            (self.eDir,''),
            (self.cAutoFlash,0),
            (self.sPaint,500),
            (self.sFill,100),
            (self.sSwell_2,2),
            (self.sErosion,2)
        )
        if self.eDir.toPlainText():
            s=self.eDir.toPlainText()
            global gWorkPath
            gWorkPath=s
            if not os.path.exists(s):
                self.eDir.setText("")
                return 
            self.displayByDir(s)

    def tool_btn_pressed(self,a):
        self.set_current_tool(a.text())
    #遍历目录找到第一个文件
    def displayByDir(self,directory):
        def warning():
            QMessageBox.warning(self, '警告', '该文件夹下没有标注图片！')
        done=False
        self.workPath = directory
        finish = utils.load_finish(self.workPath)
        global gWorkPath
        gWorkPath = self.workPath
        # 遍历所有文件
        for (root, dirs, files) in os.walk(self.workPath):
            if not files:
                warning()
                return
            for f in files:
                names=f.split('.')
                if len(names)<2: continue
                if names[1] not in ('jpg','png'): continue
                name=names[0]
                li=name.split('_')
                if(len(li)>=2 and li[1]=='gt'): # 确保不是gt文件
                    continue
                if f in finish:
                    continue
                self.workFile = directory + '/' + f
                self.processImg(self.workFile)
                self.paintFlash()
                done=True
                break
        if not done:
            warning()
    #处理图片
    def processImg(self,dir):
        self.e_filename.setText(dir)
        global LEN
        LEN=656
        src = plt.imread(dir)[:,:,:3]
        img=src
        shape=img.shape
        # ans=np.zeros(shape[:2],'uint8') #创建空白结果
        ansState=self.cb_ansState.currentText()
        if ansState=='FCN':
            ans = self.FCN.visualize(img)  #用FCN模型识别结果
            logic = ans > 60
            ans=np.zeros(ans.shape,'uint8')
            ans[logic] = 255
        elif ansState=='empty':
            ans=np.zeros(shape[:2],'uint8') #创建空白结果
        elif ansState=='keep annote':
            if type(self.data)==type(None):
                ans=np.zeros(shape[:2],'uint8') #创建空白结果
            else:
                ans=cv2.resize(self.data,shape[:2])
        elif ansState=='edge detect':
            ans=self.edge_detect(img)
        plus=self.imgPlus(img,ans)
        self.img=img
        self.plus=plus
        self.data=ans
        self.select=None
        self.undoList=[( self.data.copy(),self.plus.copy(),self.select,self.prePt)]
        self.undoPos=0  #还原点
    def edge_detect(self,src):
        global LEN
        ans = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        th_b1 = self.sB1.value()
        th_b2 = self.sB2.value()
        ans = cv2.Canny(ans, th_b1, th_b2)  # 边缘检测
        ans = np.uint8(np.absolute(ans))
        th_swell = self.sSwell.value()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (th_swell, th_swell))  # 用于膨胀
        ans = cv2.dilate(ans, kernel)  # 膨胀
        ans = np.array(ans, dtype='bool')
        if self.cFill.checkState() == Qt.Checked:
            ans = morphology.remove_small_holes(ans, self.sFill.value(), 2)
        if self.cDel.checkState() == Qt.Checked:  # self.sConnect.value()
            ans = morphology.remove_small_objects(ans, min_size=self.sConnect.value(), connectivity=2)
        ans = np.array(ans, dtype='uint8')
        self.arrReplace(ans)
        return ans
    #完成标注
    def nextImg(self):
        global LEN
        self.release_select()
        if not self.workFile: return
        row_path=self.workFile
        row_name=os.path.basename(row_path)
        f_name=row_path.split('.')[0]
        o_name=f_name+'_gt.jpg'
        imsave(o_name, cv2.resize(self.data, (LEN, LEN)))   #保存标注
        utils.dump_finish_record(self.workPath,row_name)
        self.displayByDir(self.workPath)
        return
    def saveConfigure(self,*args):
        cfg=[]
        for obj,default in args:
            if(type(obj)==type(QtWidgets.QCheckBox())):
                if obj.checkState()==Qt.Checked:
                    cfg.append(1)
                else:
                    cfg.append(0)
            elif(type(obj)==type(QtWidgets.QSpinBox())):
                cfg.append(obj.value())
            elif (type(obj) == type(QtWidgets.QTextEdit())):
                cfg.append(obj.toPlainText())
        np.save('configure.npy', cfg)

    def loadConfigure(self, *args):
        dir='configure.npy'
        isInit=False
        if not os.path.isfile(dir):
            isInit=True
            cfg=[0]*len(args)
        else:
            cfg=np.load(dir)
        index=0
        for obj,default in args:
            if isInit:cfg[index]=default
            if(type(obj)==type(QtWidgets.QCheckBox())):
                if(int(cfg[index])==1):
                    obj.setCheckState(Qt.Checked)
                else:
                    obj.setCheckState(Qt.Unchecked)
            elif(type(obj)==type(QtWidgets.QSpinBox())):
                obj.setValue(int(cfg[index]))
            elif (type(obj) == type(QtWidgets.QTextEdit())):
                obj.setText(str(cfg[index]))
            index+=1
    # 结束事件
    def closeEvent(self, event):
        self.saveConfigure(
            (self.sB1,400),
            (self.sB2,200),
            (self.sSwell,2),
            (self.sConnect,100),
            (self.cFill,1),
            (self.cDel,1),
            (self.sPen,5),
            (self.sRubber,20),
            (self.eDir,''),
            (self.cAutoFlash, 0),
            (self.sPaint, 500),
            (self.sFill, 100),
            (self.sSwell_2, 2),
            (self.sErosion, 2)
        )
    def paintEvent(self, event):
        pass
        # qp=QPainter()
        # qp.begin(self)
        # #左边的圆
        # qp.setPen(QPen(Qt.red))
        # qp.setBrush(QBrush(Qt.red,Qt.SolidPattern))
        # d=int(self.sPen.value())
        # x=195;y=580
        # qp.drawEllipse(x-d/2, y-d/2, d, d)
        # # 右边的圆
        # qp.setPen(QPen(Qt.black))
        # qp.setBrush(QBrush(Qt.black,Qt.SolidPattern))
        # d=int(self.sRubber.value())
        # x=460;y=580
        # qp.drawEllipse(x-d/2, y-d/2, d, d)
        # self.update()
        # qp.end()
    # 直线工具
    def lineTool(self,x,y):
        self.bHelp.setFocus()
        if self.prePt:
            self.addPaintDot(self.plus, (x, y), self.sPen.value(), 'rgb')
            self.addPaintDot(self.data, (x, y), self.sPen.value())
            self.addPaintDot(self.plus, self.prePt, self.sPen.value(), 'rgb')
            self.addPaintDot(self.data, self.prePt, self.sPen.value())
            self.drawLine((x, y), self.sPen.value())
            self.display()
            self.maintainUnDo((x, y))  # 维护撤销
        self.prePt = (x, y)
    # 鼠标按下
    def mousePressEvent(self, event):
        global GX
        global GY
        # GX=self.lable_img.pos().x()
        # GY=self.lable_img.pos().y()
        x=event.pos().x()-GX
        y=event.pos().y()-GY
        if event.button() ==QtCore.Qt.LeftButton:
            self.mouseIsPress=True
            #直线
            if self.toolMode=='line':
                self.lineTool(x,y)
            #油漆桶
            elif self.toolMode=='bucket':
                self.bfs1(y, x)
            #选择工具
            elif self.toolMode=='select':
                if self.data[y][x] != 0:    #选择区域有物品
                    if type(self.select)==type(None):
                        shape=self.data.shape
                        self.select=np.zeros(shape,'uint8')
                    self.bfs2(y,x)
        self.mouseIsMove=False
        self.prePt = (x, y)

    def release_select(self):
        if type(self.select) != type(None):
            logic=self.select>0
            self.data[logic]=255
            self.plus = self.imgPlus(self.img, self.data)
            shape = self.data.shape
            self.select =None
            self.paintFlash()
            self.maintainUnDo()

    # 鼠标释放
    def mouseReleaseEvent(self, event):
        if not self.toolMode=='line':
            self.prePt=None
        self.mouseIsPress = False
        self.setCursor(Qt.ArrowCursor)
        if self.mouseIsMove:
            self.maintainUnDo()#维护撤销
        if self.rubber_is_used:
            self.display()
        # self.update()
    # 鼠标移动
    def mouseMoveEvent(self, event):#鼠标移动
        global LEN
        global GX
        global GY
        x=event.pos().x()-GX
        y=event.pos().y()-GY
        self.rubber_is_used = False
        if(x<=LEN and y<=LEN and self.mouseIsPress and self.toolMode):
            self.mouseIsMove = True  # 执行了绘图或者橡皮操作
            self.setCursor(Qt.CrossCursor)
            if self.toolMode=='pen':
                self.addPaintDot(self.plus,(x,y),self.sPen.value(),'rgb')
                self.addPaintDot(self.data,(x,y),self.sPen.value())
                self.drawLine((x,y),self.sPen.value())
                self.display()
            elif self.toolMode == 'rubber':
                self.delPaintDot( (y, x), self.sRubber.value())
                self.setTmpPlus(y,x)
                self.rubber_is_used = True
            self.prePt = (x,y)
        # self.update()
        self.update()
        pass

    def setTmpPlus(self,x,y):
        img=self.plus.copy()
        bx, by = img.shape[0:2]
        pt=[x,y]
        r=0
        if self.toolMode=="pen":
            r=self.sRubber.value()
        else:r=self.sRubber.value()
        def legel(x, y):
            if x >= 0 and x < bx and y >= 0 and y < by and math.sqrt(pow(x - pt[0], 2) + pow(y - pt[1], 2)) <= r:
                return True
            else:
                return False
        for x in range(pt[0] - r, pt[0] + r + 1):
            for y in range(pt[1] - r, pt[1] + r + 1):
                if legel(x, y):
                    img[x, y, 0] = 255-img[x, y, 0]
                    img[x, y, 1] = 255-img[x, y, 1]
                    img[x, y, 2] = 255-img[x, y, 2]
            self.display(img)
    # 鼠标滚轮
    def wheelEvent(self, event):
        delta=event.angleDelta().y()
        dp=1
        dr=3
        if delta >= 120:
            if self.toolMode in ('pen','line'):
                v=self.sPen.value()+dp
                self.sPen.setValue(v)
            elif self.toolMode=='rubber':
                v=self.sRubber.value()+dr
                self.sRubber.setValue(v)
        elif delta <= -120:
            if self.toolMode in ('pen', 'line'):
                v=self.sPen.value()-dp
                self.sPen.setValue(v)
            elif self.toolMode=='rubber':
                v=self.sRubber.value()-dr
                self.sRubber.setValue(v)
    def mouseDoubleClickEvent(self, event):#双击
        pass

    def HookEvent(self,event):
        
        self.bHelp.setFocus()
        # self.setFocus()
        if self.toolMode=='select':
            if event=='KEY_UP':
                self.offsetUp()
            elif event=='KEY_DOWN':
                self.offsetDown()
            elif event=='KEY_RIGHT':
                self.offsetRight()
            elif event=='KEY_LEFT':
                self.offsetLeft()
        elif self.toolMode=='line':
            d=self.s_step.value()
            x,y=self.prePt
            self.UnDo()
            if event=='KEY_UP':
                self.lineTool(x,y-d)
            elif event=='KEY_DOWN':
                self.lineTool(x, y + d)
            elif event=='KEY_RIGHT':
                self.lineTool(x+d, y )
            elif event=='KEY_LEFT':
                self.lineTool(x -d, y)
    # 键盘按下
    def keyPressEvent(self, event):
        key=""
        if event.key() == Qt.Key_Z:
            if event.modifiers() & Qt.ControlModifier:
                key='ctrl+z'
                self.UnDo()
        elif event.key() == Qt.Key_Y:
            if event.modifiers() & Qt.ControlModifier:
                key='ctrl+y'
                self.ReDo()

        elif event.key() == Qt.Key_Escape:
            self.normalMode()
        elif event.key() == Qt.Key_Delete:
            if self.toolMode=='select' and type(self.select) != type(None):
                self.select=None
                self.paintFlash()
        elif event.key() == Qt.Key_F:
            if event.modifiers() & Qt.ControlModifier:
                self.flashImg()
        elif event.key() == Qt.Key_X:
            if event.modifiers() & Qt.ControlModifier:
                self.lookOrigin()
        elif event.key() == Qt.Key_D:
            if event.modifiers() & Qt.ControlModifier:
                self.chooseFold()
                key = ' '
        elif event.key() == Qt.Key_I:
            if event.modifiers() & Qt.ControlModifier:
                self.chooseImg()
                key = ' '
        elif event.key() == Qt.Key_A:
            if event.modifiers() & Qt.ControlModifier:
                self.abandonImg()
                key = ' '
        elif event.key() == Qt.Key_C:
            if event.modifiers() & Qt.ControlModifier:
                self.clearData()
                key = ' '
        elif event.key() == Qt.Key_V:
            if event.modifiers() & Qt.ControlModifier:
                self.lookLabeled()
                key = ' '
        elif event.key() == Qt.Key_T:
            if event.modifiers() & Qt.ControlModifier:
                self.lookTable()
                key = ' '
        elif event.key() == Qt.Key_H:
            if event.modifiers() & Qt.ControlModifier:
                self.Help()
                key = ' '
        # if event.key() == Qt.Key_Up:
        #     if event.modifiers() & Qt.ControlModifier:
        #         print('up')
        self.update()
        # if key:
        #     self.update()
        # else:
        #     QWidget.keyPressEvent(self, event)
    def keyReleaseEvent(self, event):
        if  event.key()==Qt.Key_X :
            self.pressX = False
        self.update()
            # self.update()
        QWidget.keyReleaseEvent(self, event)
    def extractDir(self,s):
        pos=s.rfind('/')
        s=s[0:pos]
        return s
    #选择文件夹
    def chooseFold(self):
        directory = QFileDialog.getExistingDirectory(self,"选取文件夹")
        if not directory :return
        self.eDir.setText(directory)
        self.displayByDir(directory)
    #选择图片
    def chooseImg(self):
        dir=os.getcwd()
        if self.workPath:
            dir=self.workPath
        fileDir, filetype = QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                     dir ,
                                    "All Files (*)")
        if not fileDir:
            return
        self.processImg(fileDir)
        self.workFile=fileDir
        self.workPath = self.extractDir(fileDir)
        self.eDir.setText(self.workPath)
        self.display()
        global gWorkPath
        gWorkPath = self.workPath
    def imgPlus(self,img,gray,color=red):#img,ans
        logic=gray>100
        ans=img.copy()
        ans[logic]=color
        return ans
        ans=a.copy()
        x,y=b.shape[:2]
        for i in range(0,x):
            for j in range(0, y):
                if b[i,j]>128:
                    ans[i, j, 0] = b[i,j];ans[i,j,1]=0;ans[i,j,2]=0
        return ans
    def arrReplace(self,arr):
        bx,by=arr.shape
        for i in range(bx):
            for j in range(by):
                if arr[i,j]==1:
                    arr[i, j]=255


    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap
    def display(self,src=None):
        self.alpha=float(self.cb_alpha.currentText())/100.
        if type(src)==type(None):
            src=self.plus.copy()
        if type(self.select)!=type(None):
            src=self.imgPlus(src,self.select,blue)
        dst=self.img.copy()
        cv2.addWeighted(dst, 1 - self.alpha, src, self.alpha, 0, dst)
        self.lable_img.setPixmap(self.img2pixmap(dst))
        # self.lable_img.setPixmap(self.img2pixmap(img))
    #画笔
    def addPaintDot(self,img,pt,r,mode='gray'):
        if (mode == 'rgb'):
            cv2.circle(img, pt, r, (255, 0, 0), -1)
        else:
            cv2.circle(img, pt, r, (255, 255, 255), -1)
    #橡皮
    def delPaintDot(self,pt,r):
        data=self.data
        img=self.img
        plus=self.plus
        bx,by=data.shape[0:2]
        def legel(x,y):
            if x>=0 and x<bx and y>=0 and y<by and math.sqrt(pow(x-pt[0],2)+pow(y-pt[1],2))<=r:
                if data[x,y]!=0:
                    return True
            return False
        for x in range(pt[0]-r,pt[0]+r+1):
            for y in range(pt[1] - r, pt[1] + r + 1):
                if legel(x,y) :
                    data[x, y] = 0
                    plus[x, y, 0] = img[x, y, 0];plus[x, y,1] = img[x, y,1];plus[x, y,2] = img[x, y,2]

    def preImg(self):
        if not self.preDir:
            QMessageBox.warning(self,"警告","没有上一张图片！")
            return
        self.initLookOrigin()
        preDir=self.workPath+'/已处理原图/'+self.preDir
        nowDir=self.workPath+'/'+self.preDir
        shutil.move(preDir,nowDir)    #移动原图
        self.workFile=nowDir
        self.processImg(self.workFile)
        self.display()
        self.preDir=None

    def flashImg(self):
        self.processImg(self.workFile)
        self.paintFlash()

    def addStateDot(self):
        self.stateMessage+='.'
        self.statusBar().showMessage(self.stateMessage)

    def usingFlash(self):
        if self.cAutoFlash.checkState()==Qt.Checked:
            self.flashImg()

    def UnDo(self):
        if self.undoPos>0: #还原点前移
            self.undoPos-=1
        else: return #记得退出
        self.data=self.undoList[self.undoPos][0].copy()
        self.plus = self.undoList[self.undoPos][1].copy()
        select=self.undoList[self.undoPos][2]
        if type(select)!=type(None):
            self.select=select.copy()
        else:self.select=select
        self.prePt=self.undoList[self.undoPos][3]
        # self.undoList=[( self.data.copy(),self.plus.copy(),self.select,self.prePt)]
        self.paintFlash()

    def ReDo(self):
        if self.undoPos+1<len(self.undoList): #还原点后移
            self.undoPos+=1
        else: return #记得退出
        self.data=self.undoList[self.undoPos][0].copy()
        self.plus = self.undoList[self.undoPos][1].copy()
        #功能未完善
        self.paintFlash()

    def maintainUnDo(self,pt=None):
        if pt==None:
            pt=self.prePt
        # 维护撤销操作代码区
        if self.undoPos + 1 < len(self.undoList):  # 开始了新的操作，删除还原点之后的数据
            # print('删除还原点之后的数据')
            del (self.undoList[self.undoPos + 1:])
        select=self.select
        if type(select)!=type(None):
            select=self.select.copy()
        else:select=None
        self.undoList.append( ( self.data.copy(),self.plus.copy(),select,pt)  )
        if len(self.undoList) > 20:  # 发现容量超标，删除第一项
            del self.undoList[0]
        self.undoPos = len(self.undoList)-1   # 更新还原点

    def drawLine(self,pt,r):
        if self.prePt==None:
            return
        cv2.line(self.plus, self.prePt, pt, (255,0,0),r*2)
        cv2.line(self.data, self.prePt, pt, (255,255,255),r*2)
        return

    def lookLabeled(self):
        global lookLabelWin
        lookLabelWin.view()
        pass
    def Help(self):
        webbrowser.open('http://www.cnblogs.com/TQCAI/p/8724862.html')
    def abandonImg(self):
        if not self.workFile: return
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "询问",
                                        "确定要放弃这张图片吗？图片将被移动到【已处理原图】文件夹",
                                        QMessageBox.Yes | QMessageBox.No)
        if(int(reply)==65536): return
        oImgDir=self.workPath+'/已处理原图'
        if(not os.path.exists(oImgDir)):
            os.makedirs(oImgDir)
        fileName=os.path.basename(self.workFile)
        oImgNewPos=oImgDir+'/'+fileName
        shutil.move(self.workFile,oImgNewPos)    #移动原图
        for (root, dirs, files) in os.walk(self.workPath):
            if not files:
                QMessageBox.warning(self,'警告','您已经完成了整个文件夹的标注，该文件夹下没有文件！')
                return
            self.workFile=root+'/'+files[0]
            break
        self.processImg(self.workFile)
        self.display()

    def clearData(self):
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "询问",
                                        "确定清除标注吗？您可以点击“刷新”重新进行自动标注",
                                        QMessageBox.Yes | QMessageBox.No)
        if (int(reply) == 65536): return
        self.data=np.zeros(self.data.shape,'uint8')
        self.select=None
        self.plus=self.img.copy()
        self.maintainUnDo()
        self.paintFlash()

    def normalMode(self):
        self.set_current_tool('pen')

    def lookTable(self):
        global keyTableWin
        keyTableWin.view()


    def paintFlash(self):
        self.sPaint.setEnabled(True)
        global LEN
        LEN=self.sPaint.value()
        if type(self.plus)!=type(np.zeros([1])):
            return
        self.plus=cv2.resize(self.plus, (LEN, LEN))
        self.data=cv2.resize(self.data, (LEN, LEN))
        self.img=cv2.resize(self.img, (LEN, LEN))
        if type(self.plus) == type(None):
            self.select=cv2.resize(self.select, (LEN, LEN))
        self.display()
        x=self.lable_img.pos().x()
        y=self.lable_img.pos().y()
        self.lable_img.setGeometry(QtCore.QRect(x, y, x+LEN,y+LEN))

    def amendData(self):
        global LEN
        LEN=500
        dir = os.getcwd()
        dataPath=self.workPath+'/标注图片'
        if os.path.exists(dataPath):
            dir = dataPath
        else:
            QMessageBox.warning(self, '警告', '您还没有完成至少一张图片的标注！')
            return
        fileDir, filetype = QFileDialog.getOpenFileName(self,
                                                        "选取文件",
                                                        dir,
                                                        "All Files (*)")
        if not fileDir:
            return
        tmp=os.path.basename(fileDir)
        name=os.path.splitext(tmp)[0]
        imgDir=self.workPath+'/标注图片/'+name+'.jpg'
        dataDir=self.workPath+'/标注数据/'+name+'.npy'
        self.data=np.load(dataDir)
        self.img=plt.imread(imgDir)
        self.plus=self.imgPlus(self.img,self.data)
        self.paintFlash()
        #移动原图
        preDir=self.workPath+'/已处理原图/'+name+'.jpg'
        nowDir=self.workPath+'/'+name+'.jpg'
        shutil.move(preDir,nowDir)    #移动原图
        self.workFile=nowDir

    def Swell(self):
        th_swell = self.sSwell_2.value()
        if th_swell<=0:return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (th_swell, th_swell))  # 用于膨胀
        self.select = cv2.dilate(self.select, kernel)  # 膨胀
        # self.plus = self.imgPlus(self.img, self.data)
        self.maintainUnDo()
        self.paintFlash()

    def Erosion(self):
        th_swell = self.sErosion.value()
        if th_swell <= 0: return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (th_swell, th_swell))  # 用于膨胀
        self.select = cv2.erode(self.select, kernel)  # 膨胀
        # self.plus = self.imgPlus(self.img, self.data)
        self.maintainUnDo()
        self.paintFlash()

    def bfs1(self,x,y):
        self.stateMessage = '正在填充，请耐心等待 '
        self.statusBar().showMessage(self.stateMessage)
        xb,yb=self.data.shape
        vis=np.zeros([xb,yb],'bool')
        def legal(tx, ty):
            if tx < xb and tx >= 0 and ty < yb and ty >= 0:
                return True
            else:
                return False
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        q = Queue()
        q.put((x, y))
        while not q.empty():
            tmp = q.get()
            cx, cy = tmp
            self.data[cx][cy] = 255
            self.plus[cx][cy][0] = 255
            self.plus[cx][cy][1] = 0
            self.plus[cx][cy][2] = 0
            vis[cx][cy] = 1
            for i in range(0, 4):
                tx = cx + dx[i]
                ty = cy + dy[i]
                if (legal(tx, ty)):
                    if vis[tx][ty] == 0 and self.data[tx][ty]<1:
                        q.put((tx, ty))
                        vis[tx][ty] = 1
        self.maintainUnDo()
        self.paintFlash()
        self.stateMessage = '填充完毕 '
        self.statusBar().showMessage(self.stateMessage)

    def bfs2(self,x,y):
        self.stateMessage = '正在选择，请耐心等待 '
        self.statusBar().showMessage(self.stateMessage)
        xb,yb=self.data.shape
        vis=np.zeros([xb,yb],'bool')
        def legal(tx, ty):
            if tx < xb and tx >= 0 and ty < yb and ty >= 0:
                return True
            else:
                return False
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        q = Queue()
        q.put((x, y))
        while not q.empty():
            tmp = q.get()
            cx, cy = tmp
            self.data[cx][cy] = 0
            self.select[cx][cy] = 255
            vis[cx][cy] = 1
            for i in range(0, 4):
                tx = cx + dx[i]
                ty = cy + dy[i]
                if (legal(tx, ty)):
                    if vis[tx][ty] == 0 and self.data[tx][ty]>60:
                        q.put((tx, ty))
                        vis[tx][ty] = 1
        self.plus=self.imgPlus(self.img,self.data)
        self.paintFlash()
        self.stateMessage = '选择完毕 '
        self.statusBar().showMessage(self.stateMessage)
        self.maintainUnDo() #维护撤销

    def fillNeighbor(self,x,y):
        threshold=self.sFill_2.value()
        org=self.img[x,y]
        self.stateMessage = '正在填充，请耐心等待 '
        self.statusBar().showMessage(self.stateMessage)
        xb,yb=self.img.shape[:2]
        vis=np.zeros([xb,yb],'bool')
        def legal(tx, ty):
            if tx < xb and tx >= 0 and ty < yb and ty >= 0:
                return True
            else:
                return False
        def inThreshold(rgb):
            if abs(rgb[0]-org[0])<threshold and abs(rgb[1]-org[1])<threshold and abs(rgb[2]-org[2])<threshold :
                return True
            return False
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        q = Queue()
        q.put((x, y))
        while not q.empty():
            tmp = q.get()
            cx, cy = tmp
            self.data[cx][cy] = 255
            self.plus[cx][cy][0] = 255
            self.plus[cx][cy][1] = 0
            self.plus[cx][cy][2] = 0
            vis[cx][cy] = 1
            for i in range(0, 4):
                tx = cx + dx[i]
                ty = cy + dy[i]
                if (legal(tx, ty) and inThreshold(self.img[tx,ty])):
                    if vis[tx][ty] == 0 and self.data[tx][ty]<1:
                        q.put((tx, ty))
                        vis[tx][ty] = 1
        self.maintainUnDo()
        self.paintFlash()
        self.stateMessage = '填充完毕 '
        self.statusBar().showMessage(self.stateMessage)

    def offsetUp(self):
        d = self.s_step.value()
        xb,yb=self.select.shape
        zeroRow=np.zeros([1,yb])
        for i in range(d):
            self.select=np.delete(self.select,0,0)#删除索引0，在0轴上
            self.select=np.row_stack((self.select,zeroRow))
        self.paintFlash()
        self.maintainUnDo() #维护撤销

    def offsetDown(self):
        d = self.s_step.value()
        xb,yb=self.select.shape
        zeroRow=np.zeros([1,yb])
        for i in range(d):
            self.select=np.delete(self.select,-1,0)#删除最后一个索引，在0轴上
            self.select=np.row_stack((zeroRow,self.select))
        self.paintFlash()
        self.maintainUnDo() #维护撤销

    def offsetLeft(self):
        d = self.s_step.value()
        xb,yb=self.select.shape
        zeroCol=np.zeros([xb,1])
        for i in range(d):
            self.select=np.delete(self.select,0,1)#删除索引0，在1轴上
            self.select=np.column_stack((self.select,zeroCol))
        self.paintFlash()
        self.maintainUnDo() #维护撤销

    def offsetRight(self):
        d = self.s_step.value()
        xb,yb=self.select.shape
        zeroCol=np.zeros([xb,1])
        for i in range(d):
            self.select=np.delete(self.select,-1,1)#删除索引0，在1轴上
            self.select=np.column_stack((zeroCol,self.select))
        self.paintFlash()
        self.maintainUnDo() #维护撤销

    def openImg(self):
        if platform.system()=='Linux':
            cmd='xdg-open'
        elif platform.system()=='Windows':
            cmd='start'
        cmd_line=cmd+' '+self.e_filename.toPlainText()
        print(cmd_line)
        os.system(cmd_line)

    def alphaChange(self,index):
        # self.cb_alpha.itemText(index)
        self.paintFlash()

    def lookOrigin(self):
        if self.cb_alpha.currentText()=='0':
            self.cb_alpha.setCurrentText('30')
        else:
            self.cb_alpha.setCurrentText('0')

class HookThread(QThread):
    trigger = pyqtSignal(str)
    def __int__(self):
        super(HookThread, self).__init__()

    def detectInputKey(self): #core function for linux
        from evdev import InputDevice
        dev = InputDevice('/dev/input/event12')
        # print(dev)
        # print(dev.capabilities(verbose=True))
        while True:
            # print('\n')
            select([dev], [], [])
            for event in dev.read():
                code=event.code
                value=event.value
                if code==0 or (code==4 and value>2 ):
                    pass
                else:
                    key=self.code2key[str(code)]
                    status=self.val2sts[value]
                    if status in('release','keep-press') and key in (self.allowedEvent):
                        self.trigger.emit(key)
                    # print(status,key)
            # print("code:%s value:%s" % (code, value))

    def handle_events(self,args): #core function for Windows
        if isinstance(args, self.KeyboardEvent):
            if args.event_type == 'key down':
                key=args.current_key.upper()
                key='KEY_'+key
                if key in self.allowedEvent:
                    self.trigger.emit(key)
            if args.current_key == 'G' and args.event_type == 'key down' and 'Lcontrol' in args.pressed_key:
                os.system("python getImgFromClipboard.py")

    def run(self):
        self.allowedEvent = ('KEY_UP', 'KEY_LEFT', 'KEY_RIGHT', 'KEY_DOWN')
        if platform.system()=='Linux':
            self.code2key = dict()
            self.val2sts = {1: 'press', 2: 'keep-press', 0: 'release'}
            self.code2key = json.load(open(keyboard_name, 'r'))
            if detectInputKeyInLinux:
                self.detectInputKey()
        elif platform.system()=='Windows':
            from pyhooked import Hook, KeyboardEvent
            self.KeyboardEvent=KeyboardEvent
            self.hk = Hook()  # make a new instance of PyHooked
            self.hk.handler = self.handle_events  # add a new shortcut ctrl+a, or triggered on mouseover of (300,400)
            self.hk.hook()  # hook into the events, and listen to the presses


if __name__=="__main__":
    app=QApplication(sys.argv)
    HookThread=HookThread()
    HookThread.start()
    myWin=MyMainWindow()
    HookThread.trigger.connect(myWin.HookEvent)
    global lookLabelWin
    global keyTableWin
    lookLabelWin=My_dLookLabeled()
    keyTableWin=My_dKeyTable()
    myWin.bLookLabeled.clicked.connect(lookLabelWin.view)
    myWin.show()
    myWin.setMouseTracking(True)
    sys.exit(app.exec())


# /home/tqc/anaconda3/envs/tf/bin/python main.py


# 打包注意加上：
# hiddenimports=["pywt","pywt._extensions._cwt"]