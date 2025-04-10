import numpy as np
import os
import torch
import tqdm
import cv2
from modules.xfeat import XFeat
import matplotlib.pyplot as plt
import random

def generate_random_color():
    """
    生成一个随机的BGR颜色
    :return: 随机颜色 (B, G, R)
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# 绘制匹配的特征点并拼接图像
def draw_matches(img1, img2, kpts1, kpts2, offset,flag,num):
    """
    绘制两张图片的匹配点和连接线
    :param img1: 第一张图片
    :param img2: 第二张图片
    :param kpts1: 第一张图片的匹配点列表
    :param kpts2: 第二张图片的匹配点列表
    :param offset: 第二张图像的偏移量
    :return: 拼接后的图像，包含匹配的关键点和连接线
    """
    if flag:
        combined_img = np.hstack((img1, img2))  # 拼接两幅图像
        for pt1, pt2 in zip(kpts1, kpts2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]) + offset, int(pt2[1])  # 第二张图的匹配点加上偏移量

            # 生成一个随机的颜色
            match_color = generate_random_color()

            # 绘制匹配点（增大点的尺寸，半径为10）
            cv2.circle(combined_img, (x1, y1), 10, match_color, -1)  # 绘制第一个图像的关键点
            cv2.circle(combined_img, (x2, y2), 10, match_color, -1)  # 绘制第二个图像的关键点
            # 绘制连接线
            cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色连接线
    else:
        combined_img = np.vstack((img1, img2))  # 拼接两幅图像
        for pt1, pt2 in zip(kpts1, kpts2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]) , int(pt2[1]) + offset # 第二张图的匹配点加上偏移量

            # 生成一个随机的颜色
            match_color = generate_random_color()

            # 绘制匹配点（增大点的尺寸，半径为10）
            cv2.circle(combined_img, (x1, y1), 10, match_color, -1)  # 绘制第一个图像的关键点
            cv2.circle(combined_img, (x2, y2), 10, match_color, -1)  # 绘制第二个图像的关键点
            # 绘制连接线
            cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色连接线

    fig, ax = plt.subplots(figsize=(10, 5))  # 设置绘图窗口大小
    ax.imshow(combined_img)  # 显示拼接后的图像
    ax.set_title("match results")
    plt.show()  # 显示图像

    cv2.imwrite(f"./results/match_result_{num[0]}_with_{num[1]}.jpg", combined_img)
    return combined_img


