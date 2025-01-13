import numpy as np
from scipy.spatial import distance as dist
def eye_aspect_ratio(face): #thresold 3.0   > 3.0为闭眼，< 3.0为睁眼
    kps = face['landmark_2d_106']
    A_left = dist.euclidean(kps[36], kps[41])
    B_left = dist.euclidean(kps[37], kps[42])
    C_left = dist.euclidean(kps[35], kps[39])
    A_right = dist.euclidean(kps[89], kps[93])
    B_right = dist.euclidean(kps[90], kps[95])
    C_right = dist.euclidean(kps[96], kps[91])
    ear_left = (A_left + B_left) / (2.0 * C_left)
    ear_right = (A_right + B_right) / (2.0 * C_right)
    
    return (ear_left + ear_right) / 2.0

def mouth_aspect_ratio(face):#thresold 0.37   > 0.37为张嘴 < 0.37为闭嘴
    kps = face['landmark_2d_106']

    A = np.linalg.norm(kps[68] - kps[59])
    B = np.linalg.norm(kps[63] - kps[65])
    C = np.linalg.norm(kps[52] - kps[61])
    
    mar = (A + B) / (2.0 * C)
    
    return mar

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
