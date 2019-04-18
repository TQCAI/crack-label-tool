

if type(self.select) == type(None):




    def processImgAlgorithm(self,src):
        global LEN
        # 调整大小
        im = Image.fromarray(src)
        if im.height > im.width:
            im = im.transpose(Image.ROTATE_90)  # 旋转 90 度角。
        (x, y) = im.size
        x_s = LEN  # define standard width
        y_s = int(y * x_s / x)  # calc height based on standard width
        tmp = np.array(im.resize((x_s, y_s), Image.ANTIALIAS))
        x_s = LEN  # define standard width
        y_s = int(y * x_s / x)  # calc height based on standard width
        img = np.array(im.resize((x_s, y_s), Image.ANTIALIAS))
        ans = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
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
        return img,ans