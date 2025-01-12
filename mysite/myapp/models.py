import json
import numpy as np
from django.db import models

class Face(models.Model):
    name = models.CharField(max_length=100)
    address = models.CharField(max_length=255)
    # 特征向量字段使用 TextField 存储 JSON 字符串
    feature_vector = models.TextField()

    def set_feature_vector(self, vector):
        # 存储特征向量为 JSON 字符串
        self.feature_vector = json.dumps(vector.tolist())

    def get_feature_vector(self):
        # 从 JSON 字符串恢复特征向量
        return np.array(json.loads(self.feature_vector))
