from django.http import StreamingHttpResponse
import cv2
import insightface
from insightface.app import FaceAnalysis
import torch

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.app = FaceAnalysis(allowed_modules=['detection','recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])      
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
       
        self.isOpenCamera   = True
        self.isOpenFace     = True     #人脸检测
        self.isOpenPoint    = False    #人脸关键点
        self.isOpenAlign    = False    #对齐
        self.isOpenLive     = False     #活体

    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.isOpenCamera:
            ret, frame = self.video.read()
            if ret:
                if self.isOpenFace:
                    faces = self.app.get(frame)
                    if len(faces) > 0:
                        for face in faces:
                            # 绘制边界框
                            bbox = face['bbox']
                            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                            # 绘制关键点
                            if self.isOpenPoint:
                                kps = face['landmark_2d_106']
                                for kp in kps:
                                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), -1)

                            # 检查是否需要对齐
                            if self.isOpenAlign:
                                    aligned_face = face['aligned']  # 提取对齐后的人脸
                                    embedding = aligned_face['embedding']
                            else:
                                embedding = face['embedding']  # 提取未对齐的特征
                                print(f"未对齐特征: {embedding}")

                                
                    
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