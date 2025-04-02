import cv2 as cv
import numpy as npq
import time



def follow_line(img_by, HEIGHT, WIDTH, area_threshold, e_threshold):
    # 定义三个检测区域（右、左、中）
    img_right = img_by[2*HEIGHT//5:3*HEIGHT//5, 0:WIDTH//3]  # 右侧区域：垂直40%-60%高度，水平0-1/3宽度
    img_left = img_by[2*HEIGHT//5:3*HEIGHT//5, 2*WIDTH//3:WIDTH]  # 左侧区域：垂直40%-60%高度，水平2/3-全宽
    img_center = img_by[2*HEIGHT//5:3*HEIGHT//5, WIDTH//3:2*WIDTH//3]  # 修正切片语法错误
    
    img_list = [img_right, img_left, img_center]  # 区域列表用于循环处理
    center_list = []  # 存储各区域检测到的中心点
    
    # 遍历三个区域进行轮廓检测
    for i in range(3):
        img = img_list[i]
        # 查找外轮廓（简单轮廓模式）
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > area_threshold:  # 过滤小面积噪声
                # 可视化处理
                cv.drawContours(img, contour, -1, (255, 0, 0), 3)  # 蓝色绘制原始轮廓
                peri = cv.arcLength(contour, True)  # 计算轮廓周长
                # 多边形逼近（精度为周长2%）
                approx = cv.approxPolyDP(contour, 0.02 * peri, True)
                # 获取包围盒参数
                x, y, w, h = cv.boundingRect(approx)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色绘制包围盒
                # 计算中心点坐标
                x_mid = x + w // 2
                y_mid = y + h // 2
                cv.circle(img, (x_mid, y_mid), 5, (0, 0, 255), -1)  # 红色绘制中心点
                center_list.append([x_mid, y_mid])  # 保存中心点坐标
    
    # 计算横向位置误差
    error1 = center_list[0][0] - center_list[1][0]  # 右区与左区中心点横向差
    error2 = center_list[1][0] - center_list[2][0]  # 左区与中区中心点横向差
    error = (error1 + error2) / 2  # 平均误差计算
    
    # 根据误差阈值返回控制信号
    if abs(error) < e_threshold:  # 误差在阈值范围内
        return 0   # 直行
    elif error > e_threshold:     # 向右偏移
        return 1   # 左转
    elif error < -e_threshold:    # 向左偏移
        return -1  # 右转

def get_signal(img_by, HEIGHT, WIDTH, h_threshold, v_threshold):
        # 定义两个检测区域
    y_start = 2*HEIGHT//5  # 中央区域Y起始 (40%高度处)
    x_start = 7*WIDTH//15  # 中央区域X起始 (46.6%宽度处)
    img_center = img_by[y_start:y_start+HEIGHT//5, x_start:x_start+WIDTH//15]  # 中央检测区
    
    y_up_start = 4*HEIGHT//5  # 上部区域Y起始 (80%高度处)
    img_upper = img_by[y_up_start, :x_start+WIDTH//15]  # 上部检测区
    
    # 核心检测逻辑
    corners = cv.cornerHarris(img_center, 2, 3, 0.04)  # 在中央区域检测角点
    corner = tuple(corners[0][::-1])  # 取第一个角点并转换坐标顺序
    
    # 坐标变换：局部坐标→全局坐标
    global_corner = (corner[0]+x_start, corner[1]+y_start)  # 计算在原图中的坐标
    
    # 特征点可视化
    cv.circle(img_by, global_corner, 5, (0,255,255), -1)  # 在原图绘制黄色标记点
    
    # 在上部区域检测轮廓
    contours, _ = cv.findContours(img_upper, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:  # 过滤小面积噪声
            # 轮廓分析（多边形逼近、边界框计算）
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02*peri, True)
            x, y, w, h = cv.boundingRect(approx)
            
            # 计算轮廓中心点
            x_mid, y_mid = x+w//2, y+h//2
            
            # 匹配条件判断
            if abs(y_mid - global_corner[1]) < h_threshold and \
                abs(x_mid - global_corner[0]) < v_threshold:
                count += 1  # 符合阈值条件时计数器递增
                time.sleep(0.5)  # 延时跳过这次检测
                
            if count % 3 == 0:  # 每3次有效检测返回信号
                return 2                    
            
        
        
        
         
        
               
        
    
    
    
    
    