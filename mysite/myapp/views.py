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
from myapp.live import eye_aspect_ratio,mouth_aspect_ratio,eyebrow_jaw_distance,nose_jaw_distance
class VideoCamera:

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.app = FaceAnalysis(allowed_modules=['detection','recognition','landmark_2d_106'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])      
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

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
        self.isOpenEyebrow  = False
        self.isOpenNose     = False

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



    def get_frame(self):
        if self.isOpenCamera:
            ret, frame = self.video.read()
            if ret:
                if self.isOpenFace:
                    faces = self.app.get(frame)
                    if len(faces) > 0:
                        for face in faces:
                            bbox = face['bbox']
                            result = self.recognize_face(face)
                            if self.isOpenAlign:
                                    aligned_frame = self.align_face(face, frame)
                                    print(result)
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
                                print(score)
                                if score > 0.25:
                                    print('睁眼')
                                else:
                                    print('闭眼')

                            if self.isOpenPoint:
                                kps = face['landmark_2d_106']
                                for kp in kps:
                                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), -1)
                            if result is not None:
                                cv2.putText(frame, result.name, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                                cv2.putText(frame, 'Stranger', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
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