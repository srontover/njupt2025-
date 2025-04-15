import cv2 as cv
import numpy as np
import time
import utilies as ut
import serial
import struct
from picamera2 import Picamera2

HEIGHT = 480
WIDTH = 640

# 串口通信参数配置（需与实际硬件匹配）
# 配置串口参数
ser = serial.Serial(
    port='/dev/ttyAMA0',   # 根据实际设备调整（ttyS0或ttyUSB0）
    baudrate=115200,
    timeout=1,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)

picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"},  # 主流分辨率与格式
    lores={"size": (WIDTH, HEIGHT), "format": "YUV420"},    # 可选低分辨率流
    display="lores",                                   # 预览使用低分辨率流
    encode="main",                                     # 编码使用主流                   # 水平翻转
    buffer_count=4,                                    # 缓冲区数量（默认3）
    queue=False                                        # 禁用帧队列（减少延迟）
)
picam2.configure(video_config)
picam2.start()

while True:
    img = picam2.capture_array("main")
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img_copy = img.copy()
    cv.imshow("orignal" , img_copy)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 1)
    img_canny = cv.Canny(img_blur, 50, 150)
    cv.imshow("Canny", img_canny)
    
    follow_return = ut.follow_line(img_canny, img_copy, HEIGHT, WIDTH, 500, 15)
    signal_return = ut.get_signal(img_canny, img_copy, HEIGHT, WIDTH, 15, 15, 500)
    cv.imshow("follow line", img_copy)
    print(follow_return)
    if follow_return[4] is not None:
        # 使用大端字节序打包数据：
        # '>' 表示大端字节序（网络字节序）
        # 'b' 表示1字节有符号整数（-128~127），用于状态值
        # 'H' 表示2字节无符号整数（0~65535），用于y坐标均值
        data = struct.pack('>bH', follow_return[4], int(follow_return[3]))
        ser.write(data)# 发送二进制数据包（总长度3字节）
        print(f'follow_ser: {data}')
    else:
        ser.write(b"none\n")
        print(f"follow_ser: {b"none\n"}")
    
        
    if signal_return == 2:
        # 发送单字节数值信号（0x02）
        ser.write(bytes([2]))# bytes([2]) 等效于 b'\x02'
        print(f"signal_return: {bytes([2])}")
        time.sleep(5)
        while True:
            adjust_return = ut.adjust_position(img_canny, img_copy, HEIGHT, WIDTH, 500)
            if adjust_return == 2:
                # 发送调整完成信号（0x02）
                ser.write(bytes([2]))
                print(f"adjust_return: {bytes([2])}")
                break
            if adjust_return[4] is not None:
                # 持续发送调整过程中的状态数据
                data = struct.pack('>bH', adjust_return[4], int(adjust_return[3]))
                ser.write(data)
                print(f"adjust_return: {data}")
            
    cv.imshow("Result", img_copy)
    
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
ser.close()