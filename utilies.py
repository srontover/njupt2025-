import cv2 as cv
import numpy as np
import time



def follow_line(img_by, img_copy, HEIGHT, WIDTH, area_threshold, e_threshold):
    """
    巡线检测核心算法
    基于三区域偏差检测法，通过计算左右区域与中间区域的水平位置偏差判断转向需求
    
    参数：
        img_by : 二值化输入图像（黑白色），用于轮廓检测
        img_copy : 用于绘制检测结果的彩色图像副本，可视化调试用
        HEIGHT/WIDTH : 图像尺寸，用于区域划分
        area_threshold : 有效轮廓的最小面积阈值，过滤噪声干扰
        e_threshold : 转向判断的误差阈值，决定转向灵敏度
        
    返回值：
        tuple: (y_left, y_center, y_right, y_mean, state)
            y_left/y_center/y_right: 左/中/右区域中心点Y坐标
            y_mean: 三区域Y坐标平均值（用于纵向位置判断）
            state: 转向指令 0:直行 | 1:右转 | -1:左转
    """
    # 区域初始化（左、中、右三区）
    x_start_right = 2*WIDTH//3  # 右侧区域起始X坐标（图像宽度2/3处）
    x_start_left = 0            # 左侧区域起始X坐标
    x_start_center = WIDTH//3   # 中间区域起始X坐标
    state = 0  # 初始状态：直行

    # 定义检测区域ROI（高度范围：40%-60%）
    img_left = img_by[:, 0:WIDTH//3]         # 左侧检测区（宽度前1/3）
    img_right = img_by[:, 2*WIDTH//3:WIDTH]  # 右侧检测区（宽度后1/3） 
    img_center = img_by[:, WIDTH//3:2*WIDTH//3]  # 中间检测区
    
    # 初始化处理参数
    img_list = [img_left, img_center, img_right]  # 按左-中-右顺序排列区域
    center_list = []  # 存储各区域中心点坐标（基于原图坐标系）
    # 各区域左上角原点坐标（用于坐标转换）
    start_list = [x_start_left, x_start_center, x_start_right]

    # 多区域轮廓处理
    for i, img in enumerate(img_list):  # 遍历左、中、右三个区域
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
                x_mid = x + w // 2  # 包围盒水平中心
                y_mid = y + h // 2  # 包围盒垂直中心
    if len(center_list) == 3:
        cv.circle(img_copy, (int(center_list[0][0]), int(center_list[0][1])), 5, (0, 0, 255), -1)  # 绘制红色左区中心点
        cv.circle(img_copy, (int(center_list[1][0]), int(center_list[1][1])), 5, (0, 255, 0), -1)  # 绘制绿色中区中心点
        cv.circle(img_copy, (int(center_list[2][0]), int(center_list[2][1])), 5, (255, 0, 0), -1)  # 绘制蓝色右区中心点

        # 决策逻辑（双偏差平均法）
        error1 = center_list[0][0] - center_list[1][0]  # 左区中心X - 中区中心X
        error2 = center_list[1][0] - center_list[2][0]  # 中区中心X - 右区中心X
        error  = (error1 + error2) // 2  # 平均偏差消除单区检测误差
        
        # 转向状态机（滞后比较避免震荡）
        if error > e_threshold:   # 整体右偏需左转修正
            state = -1
        elif error < -e_threshold: # 整体左偏需右转修正 
            state = 1
        else:                    # 偏差在阈值范围内保持直行
            state = 0
        
        # 横向位置偏差计算
        
        return (center_list[0][1], center_list[1][1], center_list[2][1], 
                int((center_list[0][1] + center_list[1][1] + center_list[2][1]) // 3), state)  # 返回三个点的Y坐标和平均值
    else:
        return (None, None, None,None, None)  # 无有效点时返回None

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
    img_upper = img_by[y_up_start:, :x_start+WIDTH//15]  # 上部检测区（单行像素，宽同中央区）

    # 中央区角点检测
    corners = cv.cornerHarris(img_center, 2, 3, 0.04)  # Harris角点检测
    corner = tuple(corners[0][::-1])  # 取第一个角点并转换坐标顺序(y,x)->(x,y)

    # 坐标系转换（局部->全局）
    global_corner = (corner[0]+x_start, corner[1]+y_start)  # 计算原图坐标
    cv.circle(img_copy, (int(global_corner[0]), int(global_corner[1])), 5, (0,255,255), -1)  # 绘制黄色特征点

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
                time.sleep(1)  # 防止重复检测
            
            # 三次有效匹配触发信号
            if count % 3 == 0:  
                return 2
            else:
                return None 
def get_signal_adjust(img_by,img_copy, HEIGHT, WIDTH, h_threshold, v_threshold, area_threshold):
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
    img_upper = img_by[y_up_start:, :x_start+WIDTH//15]  # 上部检测区（单行像素，宽同中央区）

    # 中央区角点检测
    corners = cv.cornerHarris(img_center, 2, 3, 0.04)  # Harris角点检测
    corner = tuple(corners[0][::-1])  # 取第一个角点并转换坐标顺序(y,x)->(x,y)

    # 坐标系转换（局部->全局）
    global_corner = (corner[0]+x_start, corner[1]+y_start)  # 计算原图坐标
    cv.circle(img_copy, (int(global_corner[0]), int(global_corner[1])), 5, (0,255,255), -1)  # 绘制黄色特征点

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
                time.sleep(0.2)  # 防止重复检测
            
            # 三次有效匹配触发信号
            if count % 3 == 0:  
                return 2
            else:
                return None 

def adjust_position(img_by, img_copy, HEIGHT, WIDTH, area_threshold):
    """
    车辆位置调整检测函数
    通过整合巡线检测和特殊信号检测的结果，判断当前车辆位置状态
    
    参数:
        img_by: 二值化输入图像(黑白色)
        img_copy: 用于绘制检测结果的彩色图像副本 
        HEIGHT: 图像高度
        WIDTH: 图像宽度
        area_threshold: 有效轮廓的最小面积阈值
        
    返回值:
        如果检测到特殊信号(2):
            返回信号值2
        否则:
            返回follow_line函数的全部返回值:
            y1: 左侧区域中心点Y坐标
            y2: 中间区域中心点Y坐标 
            y3: 右侧区域中心点Y坐标
            y_mean: 三个区域Y坐标的平均值
            state: 车辆当前转向状态(0直行,1右转,-1左转)
    """
    # 调用巡线检测函数获取基础位置信息
    y1, y2, y3, y_mean, state = follow_line(img_by, img_copy, HEIGHT, WIDTH, area_threshold, 15)
    
    # 检测特殊信号
    signal_return = get_signal_adjust(img_by, img_copy, HEIGHT, WIDTH, 15, 15, area_threshold)
    
    # 优先处理特殊信号
    if signal_return == 2:
        return 2
    
    # 无特殊信号时返回巡线检测结果
    return (y1, y2, y3, y_mean, state)   
        
        
         
        
               
        
    
    
    
    
    