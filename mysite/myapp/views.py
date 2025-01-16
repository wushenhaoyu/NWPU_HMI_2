import json
import os
from django.http import JsonResponse, StreamingHttpResponse
import cv2
import insightface
from insightface.app import FaceAnalysis
import torch
import numpy as np
from scipy.spatial.distance import cosine
from django.views.decorators.csrf import csrf_exempt
from myapp.models import Face
from ultralytics import YOLO
import mediapipe as mp
from myapp.live import eye_aspect_ratio,mouth_aspect_ratio,nose_jaw_distance
class VideoCamera:

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.app = FaceAnalysis(allowed_modules=['detection','recognition','landmark_2d_106'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])      
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        self.model = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.isOpenCamera   = False 
        self.isOpenFace     = False     #人脸检测
        self.isOpenPoint    = False    #人脸关键点
        self.isOpenAlign    = False    #对齐
        self.isOpenLive     = False     #活体

        self.isStorageFace = False
        self.name = ""
        self.embedding = []

        self.isOpenEye      = True
        self.isOpenMouth    = False
        self.isHead         = False

        self.EyeCount = 0
        self.MouthCount = 0
        self.HeadLeftCount = 0
        self.HeadRightCount = 0
        self.HeadShakeCount = 0

        self.EyeState = 'Open'
        self.MouthState = 'Open'
        self.HeadState = 'Straight'


        self.isHand = False
        self.isHandPoint = False

        self.frame = None
    def __del__(self):
        self.video.release()

    def align_face(self, face, frame):
        left_eye = face['kps'][0]  
        right_eye = face['kps'][1]  

        eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi 

        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 1.0 是缩放因子
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        rotated_left_eye = np.dot(M[:, :2], np.array(left_eye).T) + M[:, 2]
        rotated_right_eye = np.dot(M[:, :2], np.array(right_eye).T) + M[:, 2]

        bbox = face['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]

        target_width = 112
        target_height = 112
        scale_factor = min(target_width / face_width, target_height / face_height)
        resized_face = cv2.resize(rotated_frame, None, fx=scale_factor, fy=scale_factor)

        return resized_face

    def storage_face(self, frame, face , name):
        if not os.path.exists('database'):
            os.mkdir('database')
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox[:]
        embedding = face['embedding']
        
        # 裁剪出人脸图像
        face_image = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # 生成一个唯一的图片名称
        
        
        # 创建 Face 对象并存储信息到数据库
        face_entry = Face(name=name, address="1")
        face_entry.set_feature_vector(embedding)
        face_entry.save()

        image_filename = f"database/{face_entry.id}/{name}_{str(np.random.randint(1000))}.jpg"
        face_entry.address = image_filename
        face_entry.save()
        # 将人脸图像保存到文件
        cv2.imwrite(image_filename, face_image)

        print(f"Stored face info for {name} at {image_filename}")
        return face_entry
    
    def recognize_face(self, face, threshold=0.4):

        input_embedding = face['embedding']
        all_faces = Face.objects.all()
        min_distance = float('inf')
        recognized_face = None

        for face_entry in all_faces:

            db_embedding = face_entry.get_feature_vector()
            distance = cosine(input_embedding, db_embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_face = face_entry
        
        if recognized_face is not None and min_distance < threshold:
            #print(f"Recognized face: {recognized_face.name}, Distance: {min_distance}")
            return recognized_face  
        else:
            #print(f"No match found! (Minimum distance: {min_distance}, Threshold: {threshold})")
            return None  

    def reset_count(self):
        self.EyeCount = 0
        self.MouthCount = 0
        self.HeadLeftCount = 0
        self.HeadRightCount = 0
        self.HeadShakeCount = 0


    def get_frame(self):
        if self.isOpenCamera:
            ret, frame = self.video.read()
            self.frame = frame
            if ret:
                if self.isOpenFace:
                    faces = self.app.get(self.frame)
                    if len(faces) > 0:
                        for face in faces:
                            bbox = face['bbox']
                            result = self.recognize_face(face)
                            if self.isOpenAlign:
                                    aligned_frame = self.align_face(face, self.frame)
                                    #print(result)
                                    #cv2.imwrite('aligned_face.png', aligned_face)
                                    aligned_faces = self.app.get(aligned_frame)  
                                    #print(aligned_faces)
                                    if len(aligned_faces) == 1:
                                        if self.isStorageFace and result is None:
                                            data = self.storage_face(aligned_frame, aligned_faces[0], self.name)
                                            self.isStorageFace = False
                                            if data is not None:
                                                print(f"存储成功: {data}")
                                    else:
                                        pass
                                            
                            else:
                                embedding = face['embedding']  
                            if self.isOpenEye:
                                score = eye_aspect_ratio(face)
                                #print(score)
                                if score > 3.3  :
                                    if self.EyeState == 'Open':
                                        self.EyeCount += 1
                                    self.EyeState = 'Close'
                                else:
                                    self.EyeState = 'Open'
                            if self.isOpenMouth:
                                score = mouth_aspect_ratio(face)
                                #print(score)
                                if score > 0.37:
                                    if self.MouthState == 'Close':
                                        self.MouthCount += 1
                                    self.MouthState = 'Open'
                                else:
                                    self.MouthState = 'Close'
                            if self.isHead:
                                score = nose_jaw_distance(face)
                                face_left1, face_right1, face_left2, face_right2 = score
                                if face_left1 > face_right1 * 1.5 and face_left2 > face_right2 * 1.5:
                                    if self.HeadState != 'Left':
                                        self.HeadLeftCount += 1
                                        self.HeadState = 'Left'
                                elif face_left1  * 1.5 < face_right1 and face_left2  * 1.5 < face_right2:
                                    if self.HeadState != 'Right':
                                        self.HeadRightCount += 1
                                        self.HeadState = 'Right'
                                else:
                                    if self.HeadState != 'Straight':
                                        self.HeadShakeCount += 1
                                        self.HeadState = 'Straight'


                            if self.isOpenPoint:
                                kps = face['landmark_2d_106']
                                for kp in kps:
                                    cv2.circle(self.frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), -1)
                            if result is not None:
                                cv2.putText(self.frame, result.name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                            else:
                                cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                                cv2.putText(self.frame, 'Stranger', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if self.isHand:
                    detections = self.model(frame, stream=True)
                    for detection in detections:
                        for box in detection.boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 转换为整数
                            conf = box.conf[0]  # 置信度
                            cls = int(box.cls[0])  # 类别索引
                            label = f"{self.model.names[cls]} {conf:.2f}"  # 获取类别名称和置信度

                            # 绘制边界框和类别
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if self.isHandPoint:
                        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(frame_)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                                )
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                return jpeg.tobytes()
                    
            else:
                return None

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

camera = VideoCamera()
def video_feed(request):
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')

def turn_camera(request):
    try:
        if camera.isOpenCamera:
            camera.video.release()
            camera.isOpenCamera = False
            return JsonResponse({'status': 0, 'message': 'Camera turned successfully'})
        elif not camera.isOpenCamera:
            camera.video = cv2.VideoCapture(0)
            camera.isOpenCamera = True
            return JsonResponse({'status': 1, 'message': 'Camera turned successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def turn_face(request):
    try:
        if camera.isOpenFace:
            camera.isOpenFace = False
            return JsonResponse({'status': 0, 'message': 'Face detection turned off'})
        elif not camera.isOpenFace:
            camera.isOpenFace = True
            return JsonResponse({'status': 1, 'message': 'Face detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def turn_point(request):
    try:
        if camera.isOpenPoint:
            camera.isOpenPoint = False
            return JsonResponse({'status': 0, 'message': 'Face point turned off'})
        elif not camera.isOpenPoint:
            camera.isOpenPoint = True
            return JsonResponse({'status': 1, 'message': 'Face point turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    

def turn_align(request):
    try:
        if camera.isOpenAlign:
            camera.isOpenAlign = False
            return JsonResponse({'status': 0, 'message': 'Face align turned off'})
        elif not camera.isOpenAlign:
            camera.isOpenAlign = True
            return JsonResponse({'status': 1, 'message': 'Face align turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500) 

def turn_eye(request):
    try:
        if camera.isOpenEye:
            camera.isOpenEye = False
            return JsonResponse({'status': 0, 'message': 'Eye detection turned off'})
        elif not camera.isOpenEye:
            camera.isOpenEye = True
            return JsonResponse({'status': 1, 'message': 'Eye detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500) 
    
def turn_mouth(request):
    try:
        if camera.isOpenMouth:
            camera.isOpenMouth = False
            return JsonResponse({'status': 0, 'message': 'Mouth detection turned off'})
        elif not camera.isOpenMouth:
            camera.isOpenMouth = True
            return JsonResponse({'status': 1, 'message': 'Mouth detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500) 
    
def turn_head(request):
    try:
        if camera.isHead:
            camera.isHead = False
            return JsonResponse({'status': 0, 'message': 'Head detection turned off'})
        elif not camera.isHead:
            camera.isHead = True
            return JsonResponse({'status': 1, 'message': 'Head detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def reset_count(request):
    try:
        camera.reset_count()
        return JsonResponse({'status': 1, 'message': 'Reset count successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def get_count(request):
    try:
        EyeCount            =         camera.EyeCount
        MouthCount          =         camera.MouthCount
        HeadLeftCount       =         camera.HeadLeftCount
        HeadRightCount      =         camera.HeadRightCount
        HeadShakeCount      =         camera.HeadShakeCount
        return JsonResponse({'EyeCount': EyeCount, 'MouthCount': MouthCount, 'HeadLeftCount': HeadLeftCount, 'HeadRightCount': HeadRightCount, 'HeadShakeCount': HeadShakeCount})
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def turn_hand(request):
    try:
        if camera.isHand:
            camera.isHand = False
            return JsonResponse({'status': 0, 'message': 'Hand detection turned off'})
        elif not camera.isHand:
            camera.isHand = True
            return JsonResponse({'status': 1, 'message': 'Hand detection turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def turn_hand_point(request):
    try:
        if camera.isHandPoint:
            camera.isHandPoint = False
            return JsonResponse({'status': 0, 'message': 'Hand point turned off'})
        elif not camera.isHandPoint:
            camera.isHandPoint = True
            return JsonResponse({'status': 1, 'message': 'Hand point turned on'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
@csrf_exempt
def storage_face(request):
    try:
        data = json.loads(request.body)
        name = data.get('name')
        print(name)
        camera.name = name
        camera.isStorageFace = True
        response = JsonResponse({'status': 1, 'message': 'Storage face turned on'})
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response
    except Exception as e:
        response = JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        response["Access-Control-Allow-Origin"] = "http://localhost:9080"
        return response