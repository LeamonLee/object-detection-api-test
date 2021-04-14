import os, base64
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from tensorflow.python.saved_model import tag_constants

# dctModelConfig = {
#     "lpr": {
#         "namesPath": "./data/classes/lpr_custom.names",
#         "modelPath": "checkpoints/yolov4-custom-lpr-416"
#     }
#     # "facemask": {
#     #     "namesPath": "./data/classes/facemask_custom.names",
#     #     "modelPath": "checkpoints/yolov4-custom-facemask-416"
#     # }
# }
dctInfer = {}
imageSavePath = ""
INPUT_IMAGE_SIZE = 416
IOU = 0.45
SCORE = 0.25
VIDEO_OUTPUT_FORMAT = 'MP4V'
ROOT_PATH = ""

class ODDetector:
    
    def __init__(self, weightsPath):
        tf.keras.backend.clear_session()
        saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
        self.model = saved_model_loaded.signatures['serving_default']
        
    def predict(self, batch_data):
        pred_bbox = self.model(batch_data)    # Failed at this line
        return pred_bbox

def image_detect(classesName, imagePath):

    if classesName not in dctInfer.keys():
        return {"response": "classesName is not allowed"}

    if imagePath != "":
        original_image = cv2.imread(imagePath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)

        # pred_bbox = dctInfer[classesName].predict(batch_data)
        pred_bbox = dctInfer[classesName](batch_data)
        print("pred_bbox: ", pred_bbox)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        detectedImage = utils.draw_bbox_by_classes(original_image, pred_bbox, classesName=classesName)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_BGR2RGB)
        detectedImageFileName = imagePath.rsplit('.')[1] + '_detected' + '.' + imagePath.rsplit('.')[0]
        detectedImagePath = os.path.join(imageSavePath, detectedImageFileName)
        print("detectedImagePath: ", detectedImagePath)
        cv2.imwrite(detectedImagePath, detectedImage)
        
        try:
            with open(detectedImagePath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print("encoded_string: ", encoded_string)
            img_url = f'data:image/jpg;base64,{encoded_string}'
        except Exception as e:
            print(e)
        return img_url

    else:
        return {"response": "FileNotFoundError"}


if __name__ == "__main__":

    ROOT_PATH = os.getcwd()
    imageSavePath = os.path.join(ROOT_PATH, "static", "upload", "images")
    if os.path.exists(imageSavePath) == False:
        os.makedirs(imageSavePath)
    
    for k,v in cfg.YOLO.MYCLASSES.items(): 
        weightsPath = os.path.join(ROOT_PATH, v["modelPath"])
        tf.keras.backend.clear_session()
        saved_model_loaded = tf.saved_model.load(weightsPath, tags=[tag_constants.SERVING])
        dctInfer[k] = saved_model_loaded.signatures['serving_default']
        
        # dctInfer[k] = ODDetector(weightsPath)

    # classesName = "facemask"
    # imagePath = "./facemask1.jpg"
    
    classesName = "lpr"
    imagePath = "./car1.jpg"
    image_detect(classesName, imagePath)