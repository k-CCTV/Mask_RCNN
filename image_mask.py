import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN")  # 여기 수정하세요

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from mrcnn import water

#% matplotlib inline --> plt.show()로 대체
plt.show()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
WATER_WEIGHTS_PATH = "C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/logs/water20220528T0719/mask_rcnn_water_0029.h5"  # 여기 수정하세요
config = water.WaterConfig()
# balloon dataset 있는 경로로 수정!
WATER_DIR = 'C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/learning/data'  # 여기 수정하세요


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
# weights_path = model.find_last()
weights_path = "C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/logs/water20220528T0719/mask_rcnn_water_0029.h5"
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]

#출력
print("------------")
print(r['rois'])
np.set_printoptions(threshold = 1000,linewidth=1000)
np.savetxt('C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/roi.txt', r['rois'], fmt='%.2f', delimiter=",")
print("------------")
"""
print("------------")
np.set_printoptions(threshold = 1000,linewidth=1000)
np.savetxt('/content/drive/MyDrive/waterseg/mask.txt', r['masks'].reshape((3,-1)), fmt='%d', delimiter=",")
print("------------")
"""
#mrcnn.visualize 165줄 없애고 166,167,168 추가하기
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)