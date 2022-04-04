### 데이터셋 다운로드 (resize해서 작은 모델)
'''
%mkdir /content/yolov5/hardhat
%cd /content/yolov5/hardhat
!curl -L "http"
'''
from glob import glob

train_img_list = glob('content/yolov5/hardhat/train/images/*.jpg')
test_img_list = glob('content/yolov5/hardhat/test/images/*.jpg')
print(len(train_img_list))

#split해서 val을 만들어줌
from sklearn.model_selection import train_test_split

test_img_list, val_img_list = train_test_split(test_img_list, test_size=0.5, random_state=777)
print(len(test_img_list))

import yaml

with open('content/yolov5/hardhat/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list)+ '\n')
with open('content/yolov5/hardhat/test.txt', 'w') as f:
  f.write('\n'.join(test_img_list)+ '\n')
with open('content/yolov5/hardhat/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list)+ '\n')

#%cat content/yolov5/hardhat/data.yaml #확인 후 수정

#data.yaml 파일 수정
'''
%%writetemplate content/yolov5/hardhat/data.yaml

train: ./hardhat/train/images
test: ./hardhat/test/images
val: ./hardhat/test/images

nc: 3
names:['head','helmet','person']
'''


### 모델 구성
with open('data.yaml', 'r') as stream:
  num_classes = str(yaml.safe_load(stream)['nc'])

#%cat /content/yolov5/models/yolov5s.yaml

#custom yolov5 만듬
'''
%%writetemplate /content/yolov5/models/custom_yolov5s.yaml

#parameters
nc: {num_classes}

.....
'''

### 학습
'''
%time
%cd /content/yolov5
!python train.py --img 416 --batch 64 --epoch 50 --data ./hardhat/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name hardhat_results --cache
'''

'''
%load_ext tensorboard
%tensorboard --logdir runs
'''

Image(filename='runs/train/hardhat_results/results.png', width=1000)
Image(filename='runs/train/hardhat_results/train_batch0.png', width=1000)
Image(filename='runs/train/hardhat_results/val_batch0_label.png', width=1000)


### 검증
#!python val.py --weight runs/train/hardhat_results/weights/best.py --data ./hardhat/data.yaml --img 416 --iou 0.6 --half

#!python val.py --weight runs/train/hardhat_results/weights/best.py --data ./hardhat/data.yaml --img 416 --task test

### 추론
