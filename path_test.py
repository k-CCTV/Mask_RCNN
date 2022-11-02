import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT1 = FILE.parents[1]
ROOT2 = os.path.join(ROOT1, "yolov5")
print("------")
print(FILE)
print(ROOT)
print(ROOT1)
print(ROOT2)
if str(ROOT2) not in sys.path:
    sys.path.append(str(ROOT2))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT2, Path.cwd()))  # relative

print("------")
print(ROOT)
print("------")

weights=ROOT / 'yolov5s.pt'

print(weights)

ab_path = os.path.abspath(weights)

print("------")
print(ab_path)

MODEL_DIR = os.path.join(ROOT1, "logs")
WATER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "water20220528T0719", "mask_rcnn_water_0029.h5")
# balloon dataset 있는 경로로 수정!
WATER_DIR = os.path.join(ROOT1, "learning","data")

print("------")
print(WATER_WEIGHTS_PATH)
print(WATER_DIR)

YOLO_FILE = os.path.join(os.path.abspath(ROOT), "detect.py")
print("------")
print(YOLO_FILE)

YOLO_WEIGHTS = os.path.join(os.path.abspath(ROOT), "runs","train","guide_yolov5s_results2","weights","best.pt")
print("------")
print(YOLO_WEIGHTS)

EXEYOLO = "python " + YOLO_FILE + " --weights " + YOLO_WEIGHTS + " --source C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/videofile_masked01.avi"
print("------")
print(EXEYOLO)
"""
YOLO_FILE = os.path.join(YOLO_DIR, "detect.py")
YOLO_WEIGHTS = os.path.join(YOLO_DIR, "runs","train","guide_yolov5s_results2","weights","best.pt")
#영상 인자 장고 경로로 입력 필요
os.system("python" + YOLO_FILE + " --weights" + YOLO_WEIGHTS + " --source C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/videofile_masked01.avi")
#os.system("python C:/Users/owNer/Desktop/CCTV/cctvServer/yolov5/detect.py --weights C:/Users/owNer/Desktop/CCTV/cctvServer/yolov5/runs/train/guide_yolov5s_results2/weights/best.pt --source C:/Users/owNer/Desktop/CCTV/cctvServer/Mask_RCNN/videofile_masked01.avi")
"""