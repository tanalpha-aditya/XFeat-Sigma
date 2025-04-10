import numpy as np
import cv2
import torch
import pyramid_blend
from modules.xfeat import XFeat
import os
import matplotlib.pyplot as plt
import myplot
import cv2.cuda
import copy

xfeat = XFeat()
ENABLE_PLOT=False
# 锐化函数
def sharpen_image(img):
    # 使用一个简单的锐化内核
    kernel = np.array([[-1, -1, -1], [-1,9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


# 对比度增强函数（CLAHE）
def enhance_contrast(img):
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 应用CLAHE
    enhanced = clahe.apply(gray)

    # 转换回彩色图像
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# 图像预处理函数：锐化和对比度增强
def preprocess_image(img):
    img_sharpened = sharpen_image(img)
    img_contrast_enhanced = enhance_contrast(img_sharpened)
    return img_contrast_enhanced




def get_relative_position(index1,index2):
    match = re.search(pattern, image_paths[index1])
    i1, j1 = int(match.group(1)), int(match.group(2))
    match = re.search(pattern, image_paths[index2])
    i2, j2 = int(match.group(1)), int(match.group(2))
    if i1-i2 ==1:
        return "1 is to the right of 2"
    elif i1-i2 ==-1:
        return "1 is to the left of 2"
    elif j1-j2 ==1:
        return "1 is to the top of 2"
    elif j1-j2 ==-1:
        return "1 is to the bottom of 2"
    else:
        raise ValueError


def filter_matches(kpts1, kpts2, relative_position, threshold):
    """
    Filters matched keypoints where the x or y coordinate differences are below a given threshold.

    :param kpts1: List or NumPy array of keypoints from the first image. Shape: (N, 2)
    :param kpts2: List or NumPy array of keypoints from the second image. Shape: (N, 2)
    :param relative_position: Boolean indicating the axis to filter on.
                              True for horizontal (y-axis) matching, False for vertical (x-axis) matching.
    :param threshold: Threshold for the coordinate difference.
    :return: Tuple of filtered keypoints (filtered_kpts1, filtered_kpts2) as lists.
    """

    # Convert lists to NumPy arrays if they aren't already
    kpts1 = np.asarray(kpts1)
    kpts2 = np.asarray(kpts2)

    # Ensure that kpts1 and kpts2 have the same number of keypoints
    if kpts1.shape[0] != kpts2.shape[0]:
        raise ValueError("kpts1 and kpts2 must have the same number of keypoints.")

    # Select the axis based on relative_position
    axis = 1 if relative_position else 0

    # Compute absolute differences along the selected axis
    diffs = np.abs(kpts1[:, axis] - kpts2[:, axis])

    # Create a boolean mask where differences are below the threshold
    mask = diffs < threshold

    # Apply the mask to filter keypoints
    filtered_kpts1 = kpts1[mask].tolist()
    filtered_kpts2 = kpts2[mask].tolist()

    return filtered_kpts1, filtered_kpts2


def match_image_using_xfeat(target_img_index, source_img_index, ratio, top_k,threshold):
    """
    用xfeat模型匹配特征点
    :param target_img_index:
    :param source_img_index:
    :param ratio: 图像边界用来匹配特征点的比例
    :param top_k:
    :return: 拼接后的图像
    """
    target_img=images[target_img_index]
    source_img=images[source_img_index]
    relative_position = get_relative_position(target_img_index,source_img_index)
    h1, w1 = target_img.shape[:2]
    h2, w2 = source_img.shape[:2]
    target_img = preprocess_image(target_img)
    source_img = preprocess_image(source_img)
    if relative_position == "1 is to the right of 2":
        flag = 1
        region_tar = target_img[:, :int(ratio * w1), :]
        region_src = source_img[:, int((1 - ratio) * w2):, :]

        matches_list = xfeat.match_xfeat_star(region_tar, region_src, top_k=top_k)
        mkpts_tar, mkpts_src = matches_list[0], matches_list[1]
        mkpts_tar, mkpts_src = filter_matches(mkpts_tar, mkpts_src, flag, threshold)
        mkpts_tar = [(pt[0] , pt[1]) for pt in mkpts_tar]
        mkpts_src = [(pt[0] + int((1 - ratio) * w2), pt[1]) for pt in mkpts_src]
        if ENABLE_PLOT:
            myplot.draw_matches(source_img, target_img, mkpts_src, mkpts_tar, w2,flag,[target_img_index, source_img_index])

    elif relative_position == "1 is to the left of 2":
        flag = 1
        region_tar = target_img[:, int((1 - ratio) * w1):, :]
        region_src = source_img[:, :int(ratio * w2), :]
        matches_list = xfeat.match_xfeat_star(region_tar, region_src, top_k=top_k)
        mkpts_tar, mkpts_src = matches_list[0], matches_list[1]
        mkpts_tar, mkpts_src = filter_matches(mkpts_tar, mkpts_src, flag, threshold)
        mkpts_tar = [(pt[0] + int((1 - ratio) * w1), pt[1]) for pt in mkpts_tar]
        mkpts_src = [(pt[0], pt[1]) for pt in mkpts_src]
        if ENABLE_PLOT:
            myplot.draw_matches(target_img, source_img, mkpts_tar, mkpts_src, w1, flag,[target_img_index, source_img_index])

    elif relative_position == "1 is to the top of 2":
        flag = 0
        region_tar = target_img[int((1 - ratio) * h1):, :, :]
        region_src = source_img[:int(ratio * h2), :, :]
        matches_list = xfeat.match_xfeat_star(region_tar, region_src, top_k=top_k)
        mkpts_tar, mkpts_src = matches_list[0], matches_list[1]
        mkpts_tar, mkpts_src = filter_matches(mkpts_tar, mkpts_src, flag, threshold)
        mkpts_tar = [(pt[0] , pt[1]+ int((1 - ratio) * h1)) for pt in mkpts_tar]
        mkpts_src = [(pt[0], pt[1]) for pt in mkpts_src]
        if ENABLE_PLOT:
            myplot.draw_matches(target_img, source_img, mkpts_tar, mkpts_src, h1, flag,[target_img_index, source_img_index])

    else:
        flag = 0
        region_tar = target_img[:int(ratio * h1), :, :]
        region_src = source_img[int((1 - ratio) * h2):, :, :]
        matches_list = xfeat.match_xfeat_star(region_tar, region_src, top_k=top_k)
        mkpts_tar, mkpts_src = matches_list[0], matches_list[1]
        mkpts_tar, mkpts_src = filter_matches(mkpts_tar, mkpts_src, flag, threshold)
        mkpts_tar = [(pt[0] , pt[1]) for pt in mkpts_tar]
        mkpts_src = [(pt[0], pt[1]+ int((1 - ratio) * h2)) for pt in mkpts_src]
        if ENABLE_PLOT:
            myplot.draw_matches(source_img, target_img, mkpts_src, mkpts_tar, h2, flag,[target_img_index, source_img_index])

    # 筛选匹配点

    print(f"num of matched points: {len(mkpts_tar)}")



    return np.array(mkpts_tar, dtype=np.float32), np.array(mkpts_src, dtype=np.float32)

def get_trans(kpts_tar,kpts_src,warp_mode):
    if warp_mode:
        H, status = cv2.findHomography(kpts_src, kpts_tar, cv2.RANSAC)
        return H
    else:
        M, status = cv2.estimateAffinePartial2D(kpts_src, kpts_tar)
        return M


def weighted_average_matrices(matrices, weights):
    """
    计算加权平均矩阵
    :param matrices: 变换矩阵列表 (每个矩阵是2x3或3x3)
    :param weights: 对应每个矩阵的权重
    :return: 加权平均矩阵
    """
    # 确保权重是一个数组
    weights = np.array(weights)

    # 计算加权和
    weighted_sum = np.sum([weights[i] * matrices[i] for i in range(len(matrices))], axis=0)

    # 计算权重之和
    weight_sum = np.sum(weights)

    # 返回加权平均矩阵
    return weighted_sum / weight_sum

def expand_to_3x3(M):
    """ 将 2x3 仿射变换矩阵扩展为 3x3 """
    return np.vstack([M, [0, 0, 1]])

def get_swap_matrix(warp_mode, ratio, top_k, threshold):
    center_index = 7  # 中心图像的索引
    transformations = {}

    # 第一层：直接计算
    par=[[0.05,5000,150],[0.06, 5000, 150],[0.06, 5000, 150],[0.03, 5000, 150]]
    for i in range(len(first_layer)):
        ratio, top_k, threshol=par[i]
        idx=first_layer[i]
        kpts1, kpts2 = match_image_using_xfeat(center_index, idx, ratio, top_k, threshold)
        transformations[(idx, center_index)] = get_trans(kpts1, kpts2, warp_mode)


    # 第二层：考虑多条路径，每条路径两步

    for idx in second_layer:
        if idx == 1:
            # 两条路径：1 -> 2 -> 7 和 1 -> 6 -> 7
            kpts1_1, kpts2_1 = match_image_using_xfeat(2, 1, 0.05, top_k, threshold)
            kpts1_2, kpts2_2 = match_image_using_xfeat(6, 1, 0.11, top_k, threshold)
            trans_1_to_2 = get_trans(kpts1_1, kpts2_1, warp_mode)
            trans_1_to_6 = get_trans(kpts1_2, kpts2_2, warp_mode)

            # 扩展为 3x3 矩阵并计算矩阵乘法
            M1 =   expand_to_3x3(transformations[(2, 7)])@expand_to_3x3(trans_1_to_2)
            M2 =  expand_to_3x3(transformations[(6, 7)]) @expand_to_3x3(trans_1_to_6)
            M1 = M1[:2, :]
            M2=  M2[:2,:]

            # 取平均
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[len(kpts1_1),len(kpts1_2)])
        elif idx == 3:
            # 两条路径：3 -> 2 -> 7 和 3 -> 8 -> 7
            kpts1_1, kpts2_1 = match_image_using_xfeat(2, 3, 0.06, top_k, threshold)
            kpts1_2, kpts2_2 = match_image_using_xfeat(8, 3, 0.09, top_k, threshold)

            M1 = expand_to_3x3(transformations[(2, 7)]) @ expand_to_3x3(get_trans(kpts1_1, kpts2_1, warp_mode))
            M2 = expand_to_3x3(transformations[(8, 7)]) @ expand_to_3x3(get_trans(kpts1_2, kpts2_2, warp_mode))
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, 7)] = weighted_average_matrices([M1, M2],[len(kpts1_1),len(kpts1_2)])
        elif idx == 11:
            # 11 -> 6 -> 7
            kpts1_1, kpts2_1 = match_image_using_xfeat(6, 11, 0.04, top_k, threshold)
            kpts1_2, kpts2_2 = match_image_using_xfeat(12, 11, 0.06, top_k, 200)
            M1 = expand_to_3x3(transformations[(6, 7)]) @ expand_to_3x3(get_trans(kpts1_1, kpts2_1, warp_mode))
            M2 = expand_to_3x3(transformations[(12, 7)]) @ expand_to_3x3(get_trans(kpts1_2, kpts2_2, warp_mode))
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[len(kpts1_1),len(kpts1_2)])
        elif idx == 13:
            # 13 -> 12 -> 7
            kpts1_1, kpts2_1 = match_image_using_xfeat(12, 13, 0.086, top_k, 150)
            kpts1_2, kpts2_2 = match_image_using_xfeat(8, 13, 0.1, top_k, 250)
            M1 = expand_to_3x3(transformations[(12, 7)]) @ expand_to_3x3(get_trans(kpts1_1, kpts2_1, warp_mode))
            M2 = expand_to_3x3(transformations[(8, 7)]) @ expand_to_3x3(get_trans(kpts1_2, kpts2_2, warp_mode))
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[len(kpts1_1),len(kpts1_2)])

    # 第三层：只有一条路径，每条路径两步

    for idx in third_layer:
        if idx == 5:
            # 5 -> 6 -> 7
            kpts1, kpts2 = match_image_using_xfeat(6, 5, 0.035, top_k, 200)
            transformations[(idx, center_index)] = (expand_to_3x3(transformations[(6, 7)]) @ expand_to_3x3(get_trans(kpts1, kpts2, warp_mode)))[:2,:]
        elif idx == 9:
            # 9 -> 8 -> 7
            kpts1, kpts2 = match_image_using_xfeat(8, 9, 0.1, top_k, 250)
            transformations[(idx, center_index)] = (expand_to_3x3(transformations[(8, 7)]) @ expand_to_3x3(get_trans(kpts1, kpts2, warp_mode)))[:2,:]

    # 第四层：每个图像有三条路径，每条路径三步

    for idx in fourth_layer:
        if idx == 0:
            # 0 -> 1     or   0 -> 5
            kpts1, kpts2 = match_image_using_xfeat(1, 0, 0.13, top_k, 200)
            w1=len(kpts1)
            trans_0_to_1 = get_trans(kpts1, kpts2, warp_mode)
            kpts1, kpts2 = match_image_using_xfeat(5, 0, 0.06, top_k, 200)
            w2 = len(kpts1)
            trans_0_to_5 = get_trans(kpts1, kpts2, warp_mode)
            M1 = expand_to_3x3(transformations[(1, 7)]) @ expand_to_3x3(trans_0_to_1)
            M2=expand_to_3x3(transformations[(5, 7)]) @ expand_to_3x3(trans_0_to_5)
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[w1,w2])

        elif idx == 4:
            # 4 -> 3     or   4 -> 9
            kpts1, kpts2 = match_image_using_xfeat(3, 4, 0.1, top_k, 150)
            w1 = len(kpts1)
            trans_4_to_3 = get_trans(kpts1, kpts2, warp_mode)
            kpts1, kpts2 = match_image_using_xfeat(9, 4, 0.1, top_k, 150)
            w2= len(kpts1)
            trans_4_to_9 = get_trans(kpts1, kpts2, warp_mode)
            M1 = expand_to_3x3(transformations[(3, 7)]) @ expand_to_3x3(trans_4_to_3)
            M2=expand_to_3x3(transformations[(9, 7)]) @ expand_to_3x3(trans_4_to_9)
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[w1,w2])

        elif idx == 10:
            # 10 -> 5     or   10 -> 11
            kpts1, kpts2 = match_image_using_xfeat(5, 10, ratio, top_k, threshold)
            w1 = len(kpts1)
            trans_10_to_5 = get_trans(kpts1, kpts2, warp_mode)
            kpts1, kpts2 = match_image_using_xfeat(11, 10, 0.09, top_k, threshold)
            w2 = len(kpts1)
            trans_10_to_11 = get_trans(kpts1, kpts2, warp_mode)
            M1 = expand_to_3x3(transformations[(5, 7)]) @ expand_to_3x3(trans_10_to_5)
            M2 = expand_to_3x3(transformations[(11, 7)]) @ expand_to_3x3(trans_10_to_11)
            M1 = M1[:2, :]
            M2=  M2[:2,:]
            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[w1,w2])
        elif idx == 14:
            # 14 -> 9     or   14 -> 13
            kpts1, kpts2 = match_image_using_xfeat(9, 14, 0.05, top_k, 100)
            w1 = len(kpts1)
            trans_14_to_9 = get_trans(kpts1, kpts2, warp_mode)
            kpts1, kpts2 = match_image_using_xfeat(13, 14, 0.3, top_k, 150)
            w2 = len(kpts1)
            trans_14_to_13 = get_trans(kpts1, kpts2, warp_mode)
            M1 = expand_to_3x3(transformations[(9, 7)]) @ expand_to_3x3(trans_14_to_9)
            M2 = expand_to_3x3(transformations[(13, 7)]) @ expand_to_3x3(trans_14_to_13)
            M1 = M1[:2, :]
            M2=  M2[:2,:]

            transformations[(idx, center_index)] = weighted_average_matrices([M1, M2],[w1,w2])
    return transformations

def compute_output_size(images, transformations, base_index=7):
    """
    计算所有图像变换后所需的画布大小，确保所有图像能够在同一个画布上完整显示。
    :param images: 图像列表
    :param transformations: 仿射变换字典，存储从其他图像到基准图像7的变换矩阵
    :param base_index: 基准图像的索引，默认为7
    :return: 输出图像的宽度和高度
    """
    h,w=images[base_index].shape[:2]
    min_x, min_y, max_x, max_y = 0, 0, w, h
    for idx, img in enumerate(images):
        if idx == base_index:
            continue
        # 获取该图像的变换矩阵
        trans = transformations.get((idx, base_index))
        if trans is not None:
            h, w = img.shape[:2]
            # 对图像的4个角点进行变换，计算新的边界框
            corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
            transformed_corners = cv2.transform(np.array([corners]), trans)[0]
            min_x = min(min_x, transformed_corners[:, 0].min())
            min_y = min(min_y, transformed_corners[:, 1].min())
            max_x = max(max_x, transformed_corners[:, 0].max())
            max_y = max(max_y, transformed_corners[:, 1].max())

    # 输出图像的宽度和高度
    return int(max_x - min_x), int(max_y - min_y),-int(min_x),-int(min_y)


def swap_and_compute_overlap(images, transformations,out_width, out_height, offset_x, offset_y,base_index=7):
    """
    将所有图像按照仿射变换拼接成一个大图像，基于图像7为基准坐标系。
    :param images: 图像列表
    :param transformations: 仿射变换字典，存储从其他图像到基准图像7的变换矩阵
    :param base_index: 基准图像的索引，默认为7
    :return: 拼接后的大图像
    """
    # 计算输出图像的大小

    base_img = images[base_index]
    trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])


    dst_img = cv2.warpAffine(base_img, trans, (out_width, out_height))
    mask = dst_img.any(axis=2)
    mask_list = [False] * len(images)  # 使用布尔类型更高效
    img_list = [None] * len(images)    # 使用None初始化
    mask_list[base_index] = mask
    img_list[base_index] = dst_img
    img_added=[base_index]

    pair_matches={}

    for layer in L:

        for j in range(len(layer)):
            idx=layer[j]
            img=images[idx]
            trans = transformations.get((idx, base_index))
            if trans is not None:
                trans_copy = trans.copy()

                # 更新仿射矩阵，将offset_x, offset_y应用到平移部分
                trans_copy[0, 2] += offset_x
                trans_copy[1, 2] += offset_y
                transformed_img = cv2.warpAffine(img, trans_copy, (out_width, out_height))
                mask = transformed_img.any(axis=2)
                mask_list[idx]=mask
                img_list[idx]=transformed_img
                for added_img_idx in img_added:
                    image_now = img_list[added_img_idx]
                    # 计算重叠掩膜
                    mask_overlap = mask & mask_list[added_img_idx]
                    # 将布尔掩膜转换为 0 和 1
                    mask_overlap_int = mask_overlap.astype(np.uint8)

                    area_overlap = mask_overlap_int.sum()

                    if area_overlap == 0:
                        continue  # 无重叠区域，跳过

                    # 计算重叠区域的平均颜色
                    Iab = transformed_img[mask_overlap].reshape(-1, 3).mean(axis=0)
                    Iba = image_now[mask_overlap].reshape(-1, 3).mean(axis=0)
                    # 更新匹配字典
                    pair_matches[(idx, added_img_idx)] = {
                        "overlap_area": area_overlap,
                        "Iab": Iab,
                        "Iba": Iba
                    }
                # 添加当前图像到已添加列表
                img_added.append(idx)
    return pair_matches

def images_sitiching_using_pyramid_blending(images, transformations,out_width, out_height ,offset_x, offset_y, leveln,base_index=7):
    """
    将所有图像按照仿射变换拼接成一个大图像，基于图像7为基准坐标系。
    :param images: 图像列表
    :param transformations: 仿射变换字典，存储从其他图像到基准图像7的变换矩阵
    :param base_index: 基准图像的索引，默认为7
    :return: 拼接后的大图像
    """
    # 计算输出图像的大小

    base_img = images[base_index]
    trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

    dst_img = cv2.warpAffine(base_img, trans, (out_width, out_height))
    if ENABLE_PLOT:
        plt.subplot(151), plt.imshow(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)), plt.title('layer0')

    num_plt=2
    for layer in L:
        src_img= np.ones((out_height, out_width, 3), dtype=np.uint8) * 0
        for j in range(len(layer)):
            idx=layer[j]
            img=images[idx]
            trans = transformations.get((idx, base_index))
            if trans is not None:

                # 获取变换后的图像
                h, w = img.shape[:2]
                # 获取变换后的图像的位置
                corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
                transformed_corners = cv2.transform(np.array([corners]), trans)[0]
                min_x = transformed_corners[:, 0].min()
                min_y = transformed_corners[:, 1].min()
                max_x = transformed_corners[:, 0].max()
                max_y = transformed_corners[:, 1].max()

                # 计算变换后的图像的位置在输出图像中的起始坐标
                start_x = int(min_x) + offset_x
                start_y = int(min_y) + offset_y
                end_x = int(max_x) + offset_x
                end_y = int(max_y) + offset_y

                # 更新仿射矩阵，将offset_x, offset_y应用到平移部分
                trans[0, 2] += offset_x
                trans[1, 2] += offset_y
                transformed_img = cv2.warpAffine(img, trans, (out_width, out_height))

                src_img[start_y:end_y, start_x:end_x, :] = transformed_img[start_y:end_y, start_x:end_x, :]


        mask_now = src_img.any(axis=2)

        dst_img=pyramid_blend.pyramid_blend(dst_img, src_img, (mask_now * 255).astype(np.uint8), leveln)

        if ENABLE_PLOT:
            plt.subplot(1,5,num_plt)
            plt.imshow(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))
            plt.title(f'layer{num_plt-1}')
            num_plt+=1
    if ENABLE_PLOT:
        plt.tight_layout()
        plt.show()
    return dst_img

def get_gain_compensation(images, pair_matches, sigma_n=10, sigma_g=0.1):
    coefficients = []
    results = []
    for k, image in enumerate(images):
        coefs = [np.zeros(3) for _ in range(len(images))]
        result = np.zeros(3)
        for (a, b), match in pair_matches.items():
            overlap_area = match["overlap_area"]
            Iab = match["Iab"]  # Shape: (3,)
            Iba = match["Iba"]  # Shape: (3,)
            if a == k:
                coefs[k] +=  overlap_area* (
                    (2 * Iab ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                coefs[b] -= (
                    (2 / sigma_n ** 2) * overlap_area * Iab * Iba
                )

                result += overlap_area / sigma_g ** 2
            elif b == k:
                coefs[k] += overlap_area * (
                    (2 * Iba ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                coefs[a] -= (
                    (2 / sigma_n ** 2) * overlap_area * Iab *Iba
                )

                result += overlap_area / sigma_g ** 2
        coefficients.append(coefs)
        results.append(result)
    coefficients = np.array(coefficients)
    results = np.array(results)
    gains = np.zeros_like(results)

    for channel in range(coefficients.shape[2]):
        coefs = coefficients[:, :, channel]
        res = results[:, channel]

        try:
            gains[:, channel] = np.linalg.solve(coefs, res)  # Shape: (num_images, 3)
        except np.linalg.LinAlgError as e:
            print(f"Error solving linear system: {e}")
            return images  # Return original images if solving fails

    max_pixel_value = max(image.max() for image in images if image is not None)
    max_gain = gains.max()
    if max_gain * max_pixel_value > 255:
        gains = gains / (max_gain * max_pixel_value) * 255


    for i,gain in enumerate(gains):
        if i==2:
            gain+=np.array([-0.05,-0.05,-0.05])
        images[i] = np.clip(images[i].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        print(f"gain for image{i} is :{gain}")

    return images



if __name__ == "__main__":
    import time
    start_time = time.time()
    import re
    image_paths = []
    dic_pare={}
    first_layer = [2, 6, 8, 12]
    second_layer = [1, 3, 11, 13]
    third_layer = [5, 9]
    fourth_layer = [0, 4, 10, 14]
    L = [first_layer, second_layer, third_layer, fourth_layer]

    pattern = r"image_(\d+)_(\d+)\.jpg"
    for i in range(1, 4):  # 对于 1 到 3 的行
        for j in range(1, 6):  # 对于 1 到 5 的列
            image_paths.append(f"./image_data/image_{i}_{j}.jpg")

    images = list(cv2.imread(img_path) for img_path in image_paths)

    dic_transformations=get_swap_matrix(0, 0.05, 5000, 150)

    out_width, out_height, offset_x, offset_y = compute_output_size(images, dic_transformations, 7)
    # 拼接图像

    pair_matches= swap_and_compute_overlap(images, dic_transformations,out_width, out_height, offset_x, offset_y, base_index=7)
    images=get_gain_compensation(images, pair_matches, sigma_n=10, sigma_g=0.1)
    final_image=images_sitiching_using_pyramid_blending(images, dic_transformations,out_width, out_height ,offset_x, offset_y,2)

    # 保存拼接后的图像
    cv2.imwrite("./results/final_image_compensation2lenel..jpg", final_image)
    print("Stitching complete, saved as /results/final_image_compensation.jpg")
    end_time = time.time()

    # 计算并输出运行时间
    elapsed_time = end_time - start_time
    print(f"running time: {elapsed_time} s")