import cv2
import mediapipe as mp
from ultralytics import YOLO

# 初始化 MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 创建 Hands 对象
hands = mp_hands.Hands(
    static_image_mode=False,  # 设置为 False 以处理视频流
    max_num_hands=2,          # 最多检测 1 只手
    min_detection_confidence=0.75,  # 检测置信度
    min_tracking_confidence=0.5     # 跟踪置信度
)

# 加载自定义 YOLO 模型
model = YOLO("best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # YOLO 模型推理
    detections = model(image, stream=True)

    # 将图像转换为 RGB 格式（MediaPipe 需要）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理图像以检测手部关键点
    results = hands.process(image)

    # 将图像转换回 BGR 格式以进行绘制
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

    # 绘制 YOLO 检测框
    for detection in detections:
        for box in detection.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 转换为整数
            conf = box.conf[0]  # 置信度
            cls = int(box.cls[0])  # 类别索引
            label = f"{model.names[cls]} {conf:.2f}"  # 获取类别名称和置信度

            # 绘制边界框和类别
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('MediaPipe Hands and YOLO', image)

    # 按 'q' 键退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
