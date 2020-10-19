from .yolov5 import YOLOv5
from .yolov3 import YOLOv3

models = {"yolov3": YOLOv3,
          "yolov5": YOLOv5}


def get_joint_model(name):
    return models.get(name,None)
