import cv2
import insightface
from insightface.app import FaceAnalysis

# 初始化 FaceAnalysis
app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'landmark_2d_106'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print(app.models)
# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多台摄像头可以尝试 1, 2, ...

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
        print(face['embedding_norm'])
        # 绘制所有106个关键点
        for kp in kps:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), -1)  # -1 表示填充圆

    # 显示图像
    cv2.imshow('Face Landmarks', img)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()