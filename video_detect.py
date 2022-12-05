import os
import sys
import cv2
import numpy as np
import random
import math
import re
import time
import numpy as np
import argparse
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

#GPU 강제 사용 코드
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#비디오 입력 코드
parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='비디오 파일을 입력해 주세요')
arg = parser.parse_args()
video_arg = arg.video

# Root directory of the project
FILE = Path(__file__).resolve()
MASK_ROOT = FILE.parents[0]
ROOT_DIR = os.path.abspath(MASK_ROOT)  # 여기 수정하세요
YOLO_ROOT = FILE.parents[1]
YOLO_DIR = os.path.join(YOLO_ROOT, "yolov5")
video_url = "http://127.0.0.1:8080/media/" + video_arg

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from mrcnn import water

#% matplotlib
#inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WATER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "water20221031T1603","mask_rcnn_water_0029.h5")
config = water.WaterConfig()
# balloon dataset 있는 경로로 수정!
WATER_DIR = os.path.join(ROOT_DIR, "learning","data")


# 그대로 실행
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()
# 못먹어도 GPU!
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# 그대로 실행
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
"""
def yolov5(weights, source):
    Yolo_Dir = "C:/Users/owNer/Desktop/CCTV/cctvServer"
    sys.path.append(Yolo_Dir)
    import yolov5.detect as detect
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    detect.run(weights, source)
"""

# Load validation dataset
dataset = water.WaterDataset()
dataset.load_water(WATER_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# 그대로 실행
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#영상 인자 받아서 넣기 아마 장고의 URL 입력 예정
#capture = cv2.VideoCapture('C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/learning/data/videofile01.mp4')
capture = cv2.VideoCapture(video_url)
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter(video_arg, codec, 24.0, size)
get_roi = False
river_mask_roi = None
count_num = 0

while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        # verbose =1 이면 출력 0이면 출력 x
        ax = get_ax(1)
        r = results[0]
        frame = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                            dataset.class_names, r['scores'], ax=ax,
                                            title="Predictions")

        """
        if get_roi == False:
            get_roi = True
            river_mask_roi = r['rois']
            np.set_printoptions(threshold=1000, linewidth=1000)
            #상대 경로로 수정 필요
            np.savetxt('C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/roi.txt', r['rois'], fmt='%d', delimiter=",")
        """
        save_roi = r['rois'].astype(int)
        if len(save_roi) > 0:
            save_rois = np.concatenate(save_roi).tolist()
            with open(MASK_ROOT / 'roi.txt', 'a') as text_f:
                text_f.write(str(save_rois))
        #강 여러개일때 고려
        with open(MASK_ROOT / 'roi.txt', 'a') as text_f:
            text_f.write("\n")
        # frame = frame.astype(np.uint8)
        output.write(frame)
        # cv2.imshow('frame', frame)
        ax = plt.clf()
        count_num = count_num + 1
        #print(str(count_num) + "   " + str(save_roi))
        print(str(count_num) + "  " + str(save_rois))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Done")
        break

capture.release()
output.release()
cv2.destroyAllWindows()

YOLO_FILE = os.path.join(YOLO_DIR, "detect.py")
YOLO_WEIGHTS = os.path.join(YOLO_DIR, "runs", "train", "guide_yolov5s_results4", "weights", "best.pt")
VIDEO_FILE = os.path.join(YOLO_ROOT, video_arg)
EXE_YOLO = "python " + YOLO_FILE + " --weights " + YOLO_WEIGHTS + " --source " + VIDEO_FILE
#yolov5("C:/Users/owNer/Desktop/CCTV/cctvServer/yolov5/runs/train/guide_yolov5s_results2/weights/best.pt","C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/videofile_masked01.avi")
os.system(EXE_YOLO)


