#모델 다운로드
from typing import Text
#!git clone https://github.com/ultralytics/yolov5
#%cd yolov5
#%pip install --qr requirements.Text

import torch
from IPython.display import Image, clear

#추론
#!python detect.py --weights  yolov5s.pt --img 640 --conf 0.25 --source data/images/