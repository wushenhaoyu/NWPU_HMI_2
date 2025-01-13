import numpy as np
from scipy.spatial import distance as dist
def eye_aspect_ratio(face):
    kps = face['landmark_2d_106']
    # 计算左眼的两组垂直关键点之间的欧式距离
    A_left = dist.euclidean(kps[37], kps[41])
    B_left = dist.euclidean(kps[38], kps[40])
    # 计算左眼的一组水平关键点之间的欧式距离
    C_left = dist.euclidean(kps[36], kps[39])
    
    # 计算右眼的两组垂直关键点之间的欧式距离
    A_right = dist.euclidean(kps[89], kps[93])
    B_right = dist.euclidean(kps[90], kps[92])
    # 计算右眼的一组水平关键点之间的欧式距离
    C_right = dist.euclidean(kps[88], kps[91])
    
    # 计算左右眼的纵横比
    ear_left = (A_left + B_left) / (2.0 * C_left)
    ear_right = (A_right + B_right) / (2.0 * C_right)
    
    # 返回左右眼的平均纵横比
    return (ear_left + ear_right) / 2.0

def mouth_aspect_ratio(face):
    kps = face['landmark_2d_106']
    # 计算嘴巴的两组垂直关键点之间的欧式距离
    A = np.linalg.norm(kps[61] - kps[67])
    B = np.linalg.norm(kps[63] - kps[65])
    # 计算嘴巴的一组水平关键点之间的欧式距离
    C = np.linalg.norm(kps[60] - kps[64])
    
    # 计算嘴巴纵横比
    mar = (A + B) / (2.0 * C)
    
    return mar

def nose_jaw_distance(face):
    kps = face['landmark_2d_106']
    # 计算鼻子上一点到左右脸边界的欧式距离
    face_left1 = dist.euclidean(kps[27], kps[0])
    face_right1 = dist.euclidean(kps[27], kps[33])
    # 计算鼻子上另一点到左右脸边界的欧式距离
    face_left2 = dist.euclidean(kps[30], kps[2])
    face_right2 = dist.euclidean(kps[30], kps[31])
    
    # 创建元组，用以保存4个欧式距离值
    face_distance = (face_left1, face_right1, face_left2, face_right2)
    
    return face_distance

def eyebrow_jaw_distance(face):
    kps = face['landmark_2d_106']
    # 计算左眉毛上一点到左右脸边界的欧式距离
    eyebrow_left = dist.euclidean(kps[24], kps[0])
    eyebrow_right = dist.euclidean(kps[24], kps[33])
    # 计算左右脸边界之间的欧式距离
    left_right = dist.euclidean(kps[0], kps[33])
    
    # 创建元组，用以保存3个欧式距离值
    eyebrow_distance = (eyebrow_left, eyebrow_right, left_right)
    
    return eyebrow_distance