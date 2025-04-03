import cv2 as cv
import numpy as np
import time



def follow_line(img_by, img_copy, HEIGHT, WIDTH, area_threshold, e_threshold):
    """
    巡线检测核心算法
    参数：
        img_by : 二值化输入图像（黑白色）
        img_copy : 用于绘制检测结果的彩色图像副本
        HEIGHT/WIDTH : 图像尺寸
        area_threshold : 有效轮廓的最小面积阈值
        e_threshold : 转向判断的误差阈值
    返回值：
        0: 直行 | 1: 左转 | -1: 右转
    """
    # 区域初始化（左、中、右三区）
    x_start_right = 2*WIDTH//3  # 右侧区域起始X坐标（图像宽度2/3处）
    x_start_left = 0            # 左侧区域起始X坐标
    x_start_center = WIDTH//3   # 中间区域起始X坐标
    y_start = 2*HEIGHT//5        # 统一垂直起始位置（图像高度40%处）

    # 定义检测区域ROI（高度范围：40%-60%）
    img_left = img_by[2*HEIGHT//5:3*HEIGHT//5, 0:WIDTH//3]         # 左侧检测区（宽度前1/3）
    img_right = img_by[2*HEIGHT//5:3*HEIGHT//5, 2*WIDTH//3:WIDTH]  # 右侧检测区（宽度后1/3） 
    img_center = img_by[2*HEIGHT//5:3*HEIGHT//5, WIDTH//3:2*WIDTH//3]  # 中间检测区
    
    # 初始化处理参数
    img_list = [img_left, img_center, img_right]  # 按左-中-右顺序排列区域
    center_list = []  # 存储各区域中心点坐标（基于原图坐标系）
    # 各区域左上角原点坐标（用于坐标转换）
    start_list = [(x_start_left, y_start), (x_start_center, y_start), (x_start_right, y_start)]

    # 多区域轮廓处理
    for i in range(3):
        img = img_list[i]
        # 查找外部轮廓（仅检测最外层轮廓）
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > area_threshold:  # 过滤噪声产生的细小轮廓
                # 轮廓多边形近似（减少轮廓点数）
                peri = cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, 0.02 * peri, True)
                
                # 获取轮廓包围盒参数
                x, y, w, h = cv.boundingRect(approx)
                
                # 计算轮廓中心点（相对于检测区域局部坐标系）
                x_mid = x + w // 2
                y_mid = y + h // 2
                
                # 转换到原图坐标系并绘制
                global_x = start_list[i][0] + x_mid  # X全局坐标 = 区域起始X + 局部X
                global_y = start_list[i][1] + y_mid  # Y全局坐标 = 区域起始Y + 局部Y
                cv.circle(img_copy, (global_x, global_y), 5, (0, 0, 255), -1)  # 红色标记点
                center_list.append([global_x, global_y])  # 记录全局坐标

    # 横向位置偏差计算
    error1 = center_list[0][1] - center_list[1][1]  # 左区与中区中心点纵向差
    error2 = center_list[1][1] - center_list[2][1]  # 中区与右区中心点纵向差
    error = (error1 + error2) / 2  # 平均偏差值

    # 控制逻辑判断（在小车朝图像向左前进的情况下）
    if abs(error) < e_threshold:  # 偏差在允许范围内
        return 0   # 维持直行
    elif error > e_threshold:     # 整体向左偏移（需要右转修正）
        return 1   
    elif error < -e_threshold:    # 整体向右偏移（需要左转修正）
        return -1  

def get_signal(img_by,img_copy, HEIGHT, WIDTH, h_threshold, v_threshold, area_threshold):
    """特殊信号检测函数
    参数：
        img_by       : 二值化输入图像
        img_copy     : 用于绘制检测结果的彩色图像副本
        HEIGHT/WIDTH : 图像尺寸
        h_threshold  : 垂直方向匹配阈值
        v_threshold  : 水平方向匹配阈值
    返回值：
        int: 当满足匹配条件时返回信号2
    """
    # 区域定义（中央特征点检测区 + 上部验证区）
    y_start = 2*HEIGHT//5  # 中央区域垂直起始（图像高度40%处）
    x_start = 7*WIDTH//15  # 中央区域水平起始（图像宽度46.6%处）
    img_center = img_by[y_start:y_start+HEIGHT//5, x_start:x_start+WIDTH//15]  # 中央检测区（高1/5，宽1/15）
    
    # 上部验证区参数
    y_up_start = 4*HEIGHT//5  # 上部区域垂直起始（图像高度80%处）
    img_upper = img_by[y_up_start, :x_start+WIDTH//15]  # 上部检测区（单行像素，宽同中央区）

    # 中央区角点检测
    corners = cv.cornerHarris(img_center, 2, 3, 0.04)  # Harris角点检测
    corner = tuple(corners[0][::-1])  # 取第一个角点并转换坐标顺序(y,x)->(x,y)

    # 坐标系转换（局部->全局）
    global_corner = (corner[0]+x_start, corner[1]+y_start)  # 计算原图坐标
    cv.circle(img_copy, global_corner, 5, (0,255,255), -1)  # 绘制黄色特征点

    # 上部区域轮廓验证
    count = 0  # 有效匹配计数器
    contours, _ = cv.findContours(img_upper, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > area_threshold:  # 过滤无效小轮廓
            # 轮廓近似处理
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02*peri, True)
            x, y, w, h = cv.boundingRect(approx)
            
            # 计算轮廓中心
            x_mid, y_mid = x+w//2, y+h//2
            
            # 双阈值匹配验证
            if abs(y_mid - global_corner[1]) < h_threshold and \
                abs(x_mid - global_corner[0]) < v_threshold:
                count += 1  # 符合条件时计数
                time.sleep(0.5)  # 防止重复检测
            
            # 三次有效匹配触发信号
            if count % 3 == 0:  
                return 2                    
            
        
        
        
         
        
               
        
    
    
    
    
    