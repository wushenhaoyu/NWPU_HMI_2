import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from scipy.spatial import distance as dist

# 初始化 FaceAnalysis
app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print(app.models)
# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多台摄像头可以尝试 1, 2, ...


def nose_jaw_distance(face):
    kps = face['landmark_2d_106']
    # 计算鼻子上一点到左右脸边界的欧式距离
    face_width = dist.euclidean(kps[1], kps[17]) 
    face_left1 = dist.euclidean(kps[72], kps[1]) / face_width
    face_right1 = dist.euclidean(kps[72], kps[17]) / face_width
    # 计算鼻子上另一点到左右脸边界的欧式距离
    face_left2 = dist.euclidean(kps[86], kps[1]) / face_width
    face_right2 = dist.euclidean(kps[86], kps[17]) / face_width
    
    # 创建元组，用以保存4个欧式距离值
    face_distance = (face_left1, face_right1, face_left2, face_right2)
    
    return face_distance

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧图像
    ret, img = cap.read()
    if not ret:
        print("无法接收帧（流结束？）")
        break

    # 检测人脸和关键点
    faces = app.get(img)

    # 绘制检测结果
    for face in faces:
        bbox = face['bbox']
        kps = face['landmark_2d_106']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        #for i in range(len(kps)):
        #    cv2.circle(img, (int(kps[i][0]), int(kps[i][1])), 1, (0, 0, 255), -1)
        #    cv2.putText(img, str(i), (int(kps[i][0]), int(kps[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        score = nose_jaw_distance(face)
        print(score)
        face_left1, face_right1, face_left2, face_right2 = score
        if face_left1 > face_right1 * 1.5 and face_left2 > face_right2 * 1.5:
            print('左转')
        elif face_left1  * 1.5 < face_right1 and face_left2  * 1.5 < face_right2:
            print('右转')
        else:
            print('straight')
        #print(eye_aspect_ratio(face))
        # 绘制所有106个关键点
        #cv2.circle(img, (int(kps[37][0]), int(kps[37][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
        #cv2.circle(img, (int(kps[38][0]), int(kps[38][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
        #cv2.circle(img, (int(kps[38][0]), int(kps[38][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
        #cv2.circle(img, (int(kps[40][0]), int(kps[40][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
        #cv2.circle(img, (int(kps[36][0]), int(kps[36][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
        #cv2.circle(img, (int(kps[39][0]), int(kps[39][1])), 1, (0, 0, 255), -1)  # -1 表示填充圆
    
    # 显示图像
    cv2.imshow('Face Landmarks', img)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()