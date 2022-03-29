###데이터셋 준비
'''
%mkdir /content/yolov5/smoke
%cd /content/yolov5/smoke
!curl -L https://
'''

from glob import glob

train_img_list = glob('content/yolov5/smoke/train/images/*.jpg')
test_img_list = glob('content/yolov5/smoke/test/images/*.jpg')
valid_img_list = glob('content/yolov5/smoke/valid/images/*.jpg')

print(len(train_img_list))

#text파일에 리스트 저장
import yaml

with open('/content/yolov5/smoke/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list)+'\n')
with open('/content/yolov5/smoke/test.txt', 'w') as f:
  f.write('\n'.join(test_img_list)+'\n')
with open('/content/yolov5/smoke/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list)+'\n')

#해당 경로를 변경해서 넣어줌
'''
%%writetemplate /content/yolov5/smoke/data.yaml

train: ./smoke/train/images
test: ./smoke/test/images
val: ./smoke/valid/images

nc: 1
names:['smoke']
'''

# %cat /content/yolov5/smoke/data.yaml


### 모델 구성
import yaml

with open('content/yolov5/smoke/data.yaml', 'r') as stream:
  num_classes = str(yaml.safe_load(stream)['nc'])   #설정한 부분을 읽어

# %cat /content/yolov5/models/yolov5s.yaml

'''
#custom을 하나 만듬
%%writetemplate /content/yolov5/models/custom_yolov5.yaml

#parameters
nc: {num_calsses}
........

#backbone
'''


### 학습
'''
%%time
%cd /content/yolov5
!python train.py --img 640 --batch 32 --epochs 100 --data ./smoke/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name smoke_results --cache
'''

#텐서보드
'''
%load_ext tensorboard
%tensorboard --logdir runs
'''

Image(filename='runs/train/smoke_results/results.png', width=100)
Image(filename='runs/train/smoke_results/train_batch0.jpg', width=100)  #위치만 detection
Image(filename='runs/train/smoke_results/val_batch0_label.png', width=100)  #레이블 이름까지 추가

###검증
#!python val.py --weights runs/train/smoke_results/weights/best.pt --data ./smoke/data.yaml --img 640 --iou 0.65 --half

#테스트
#!python val.py --weights runs/train/smoke_results/weights/best.pt --data ./smoke/data.yaml --img 640 --task test