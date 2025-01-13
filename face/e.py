import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 创建 Hands 对象
hands = mp_hands.Hands(
    static_image_mode=False,  # 设置为 False 以处理视频流
    max_num_hands=2,  # 最多检测 2 只手
    min_detection_confidence=0.75,  # 检测置信度
    min_tracking_confidence=0.5  # 跟踪置信度
)

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 将图像转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理图像以检测手部关键点
    results = hands.process(image)

    # 将图像转换回 BGR 格式以进行绘制
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 检查是否检测到手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点及其连接线
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

    # 显示图像
    cv2.imshow('MediaPipe Hands', image)

    # 按 'q' 键退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()   