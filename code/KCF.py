from numpy.fft import fft2, ifft2, fftshift
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class Hog_feature():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
    #   获取hog向量和图片
    def extract(self):
        # 获得原图的shape
        height, width = self.img.shape
        # 计算原图的梯度大小
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        # cell_gradient_vector用来保存每个细胞的梯度向量
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        # 大小等于划分成细胞之后的图
        height_cell,width_cell,_ = np.shape(cell_gradient_vector)
        #   计算每个细胞的梯度直方图
        for i in range(height_cell):
            for j in range(width_cell):
                # 分割出这个细胞的梯度 切片
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                # 分割出这个细胞的角度
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 使用函数cell_gradient蒋每个细胞内的角度和梯度转化为梯度直方图格式
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # hog图像
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        # 画图，用一个np.zeros([height, width]和原图一样大，来画cell_gradient_vector
        hog_vector = []
        # block为2x2
        for i in range(height_cell - 1):
            for j in range(width_cell - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                print(block_vector)
                hog_vector.append(block_vector)
        return hog_vector, hog_image
    #   计算原图的梯度大小
    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle
    #   分解角度信息到不同角度的直方图上
    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers
    #   计算每个像素点所属的角度
    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod
    #   将梯度直方图进行绘图
    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

class HOG:
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nBins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nBins)

    def get_feature(self, image):
        winStride = self.winSize
        w, h = self.winSize
        w_block, h_block = self.blockStride
        w = w//w_block - 1
        h = h//h_block - 1
        # 计算给定图像的HOG特征描述子，一个n*1的特征向量
        hist = self.hog.compute(img=image, winStride=winStride, padding=(0, 0))
        return hist.reshape(w, h, 36).transpose(2, 1, 0)    # 交换轴的顺序

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h*grid, w*grid))

        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(x=np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(x=np.pi / 9 * k))
                    cv2.rectangle(img=img, pt1=(j*grid, i*grid), pt2=((j + 1) * grid, (i + 1) * grid),
                                  color=(255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=1)
        cv2.imshow("img", img)
        cv2.waitKey(0)

class Tracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

        # 算法变量定义
        self.scale_h = 0.
        self.scale_w = 0.

        self.ph = 0
        self.pw = 0
        self.hog = HOG((self.pw, self.pw))

        self.alphaf = None
        self.x = None
        self.roi = None

    def first_frame(self, image, roi):
        """
        对视频的第一帧进行标记，更新tracer的参数
        :param image: 第一帧图像
        :param roi: 第一帧图像的初始ROI元组
        :return: None
        """
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        # 确定Patch的大小，并在此Patch中提取HOG特征描述子
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        # 在矩形框的中心采样、提取特征
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])

        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        对给定的图像，重新计算其目标的位置
        :param image:
        :return:
        """
        # 包含矩形框信息的四元组(min_x, min_y, w, h)
        cx, cy, w, h = self.roi
        max_response = -1   # 最大响应值

        for scale in [0.95, 1.0, 1.05]:
            # 将ROI值处理为整数
            roi = map(int, (cx, cy, w * scale, h * scale))

            z = self.get_feature(image, roi)    # tuple(36, h, w)
            # 计算响应
            responses = self.detect(self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z

        # 更新矩形框的相关参数
        self.roi = (cx + dx, cy + dy, best_w, best_h)

        # 更新模板
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        """
        对特征进行采样
        :param image:
        :param roi: 包含矩形框信息的四元组(min_x, min_y, w, h)
        :return:
        """
        # 对矩形框做2.5倍的Padding处理
        cx, cy, w, h = roi
        w = int(w*self.padding)//2*2
        h = int(h*self.padding)//2*2
        x = int(cx - w//2)
        y = int(cy - h//2)

        # 矩形框所覆盖的距离
        sub_img = image[y:y+h, x:x+w, :]
        resized_img = cv2.resize(src=sub_img, dsize=(self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw)/255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_img)
            if self.debug:
                self.hog.show_hog(feature)

        # Hog特征的通道数、高估、宽度
        fc, fh, fw = feature.shape
        self.scale_h = float(fh)/h
        self.scale_w = float(fw)/w

        # 两个二维数组，前者(fh，1)，后者(1，fw)
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))

        # 一个fh x fw的矩阵
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def gaussian_peak(self, w, h):
        """
        :param w:
        :param h:
        :return:      一个w*h的高斯矩阵
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h//2, w//2

        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def kernel_correlation(self, x1, x2, sigma):
        """
        核化的相关滤波操作
        :param x1:
        :param x2:
        :param sigma:   高斯参数sigma
        :return:
        """
        # 转换到傅里叶空间
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        # \hat{x^*} \otimes \hat{x}'，x*的共轭转置与x'的乘积
        tmp = np.conj(fx1) * fx2
        # 离散傅里叶逆变换转换回真实空间
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        # 将零频率分量移到频谱中心。
        idft_rbf = fftshift(idft_rbf)

        # 高斯核的径向基函数
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        print(k.shape)
        return k

    def train(self, x, y, sigma, lambdar):
        """
        原文所给参考train函数
        :param x:
        :param y:
        :param sigma:
        :param lambdar:
        :return:
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, x, z, sigma):
        """
        原文所给参考detect函数
        :param x:
        :param z:
        :param sigma:
        :return:
        """
        k = self.kernel_correlation(x, z, sigma)
        # 傅里叶逆变换的实部
        return np.real(ifft2(self.alphaf * fft2(k)))

if __name__ == '__main__':
    img = cv2.imread("../data/Vector_Tsoi.jpg", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    hog = Hog_feature(img, cell_size=20, bin_size=12)
    vector, image = hog.extract()
    gradient_magnitude, gradient_angle = hog.global_gradient()
    print(gradient_magnitude.shape)
    print(len(vector))
    print(len(vector[0]))
    print(image.shape)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()