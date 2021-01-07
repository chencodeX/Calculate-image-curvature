#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen
'''
@Author : zihao.chen
@File : caml_curvature.py 
@Create Date : 2021/1/5
@Descirption :
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


def PJcurvature(x, y):
    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2. + b[1] ** 2.)


def getCurvature(vecContourPoints, step):
    # 生成一个空列表，等下存曲率
    vecCurvature = [None] * vecContourPoints.shape[0]
    if len(vecContourPoints) < step:
        return vecContourPoints
    # 这一步是判断坐标点是不是头尾相连的 （Close）
    frontToBack = vecContourPoints[0] - vecContourPoints[-1]
    isClose = int(max(abs(frontToBack[0][0]), abs(frontToBack[0][1]))) <= 1
    for i in range(len(vecContourPoints)):
        pos = vecContourPoints[i]
        maxStep = step
        # 这个if其实这里不会触发
        if (not isClose):
            maxStep = min(min(step, i), (int)(vecContourPoints.shape[0] - 1 - i))
            if maxStep == 0:
                vecCurvature[i] = np.inf
                continue
        # 取前后点索引
        iminus = i - maxStep
        iplus = i + maxStep
        # 取前后点的值
        pminus = vecContourPoints[iminus]
        pplus = vecContourPoints[iplus - len(vecContourPoints) if iplus >= len(vecContourPoints) else iplus]

        # 构造三个点的 x y的坐标列表
        x = [pminus[0][0], pos[0][0], pplus[0][0]]
        y = [pminus[0][1], pos[0][1], pplus[0][1]]

        # 算曲率
        curvature2D = abs(PJcurvature(y, x)[0])
        vecCurvature[i] = curvature2D
    return vecCurvature


img = cv2.imread('WechatIMG450.png', 0)
cv2.imwrite('beisaier/gray_img.png', img)
# 均值滤波去除噪点
median_img = cv2.medianBlur(img, 15)
cv2.imwrite('beisaier/median_img.png', median_img)

# 图像插值缩放到原来的1/16，原图分辨率较高，实际上用不到这么高清，主要是为了减少计算量
mini_img = cv2.resize(median_img, (1500, 842), interpolation=cv2.INTER_NEAREST)

# 自适应阈值二值化，提取前景背景信息
srcGray_bin = cv2.adaptiveThreshold(mini_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, -5)
cv2.imwrite('beisaier/adaptThreshold_bin.png', srcGray_bin)

# 构造一个新的图片，把核心部分抠出来放进去
srcGray_bin_crop = np.zeros_like(srcGray_bin)
srcGray_bin_crop[200:600, 500:1000] = srcGray_bin[200:600, 500:1000]
cv2.imwrite('beisaier/adaptThreshold_bin_crop.png', srcGray_bin_crop)

# 前景轮廓查找，获得所有闭合前景区域的外包坐标点。
contours = cv2.findContours(srcGray_bin_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 取坐标点最多的坐标集合，也就是最大的前景图像
max_count = -1
contour = None
for i in range(len(contours[0])):
    # print(contours[0][i].shape)
    if contours[0][i].shape[0] > max_count:
        max_count = contours[0][i].shape[0]
        contour = contours[0][i]

# 取前后点的步长
step = 3

# 根据每个坐标点的位置，和前后step个点的位置，来计算曲率
curvatures = getCurvature(contour, step)
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
print(contour.shape)
print(np.array(curvatures).shape)

fig, ax = plt.subplots()
ax.scatter(contour[:, :, 0], contour[:, :, 1], s=np.array(curvatures) * 50)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()
# ax.plot_surface(contour[:,:,0], contour[:,:,0], np.array(curvatures), rstride=1, cstride=1, cmap=cm.viridis)
# plt.p(contour[:,:,0], contour[:,:,0],)
# plt.show()
