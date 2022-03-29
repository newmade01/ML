###데이터셋정리
from glob import glob

train_img_list = glob('content/yolov5/pothole/train/images/*.jpg')
test_img_list = glob('content/yolov5/pothole/test/images/*.jpg')
valid_img_list = glob('content/yolov5/pothole/valid/images/*.jpg')

### text파일로 이미지 리스트 파일 저장
import yaml

with open('content/yolov5/pothole/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('content/yolov5/pothole/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')

with open('content/yolov5/pothole/valid.txt', 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')


#이미지의 train, test, valid 리스트를 .yaml파일에 저장
from IPython.core.magic import register_line_cell_magic
@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
'''
%%writetemplate /content/yolov5/pothole/data.yaml

train: ./pothole/train/images
test: ./pothole/test/images
valid: ./pothole/valid/images

nc: 1
names: ['pothole']
'''

###모델 구성

import yaml
with open('/content/yolov5/pothole/data.yaml', 'r') as stream:
  num_classes = str(yaml.safe_load(stream)['nc']) #number classes

#%cat /content/yolov5/models/yolov5s.yaml

#%%writetemplate /content/yolov5/models/custom_yolov5s.yaml #새로운 custom 버전을 변경

###학습
'''
학습(Training)
img: 입력 이미지 크기 정의
batch: 배치 크기 결정
epochs: 학습 기간 개수 정의
data: yaml 파일 경로 #train, test, val의 이름
cfg: 모델 구성 지정
weights: 가중치에 대한 경로 지정
name: 결과 이름
nosave: 최종 체크포인트만 저장
cache: 빠른 학습을 위한 이미지 캐시
'''

'''
%%time
%cd /content/yolov5/
!python train.py --img 640 --batch 32 --epoch 100 --data ./pothole/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name pothole_result --cache 
'''

'''
%load_ext tensorboard
%tensorboard --logdir runs
'''

Imaage(filename='/content/yolov5/runs/train/pothole_results/result.png', width=1000)    #결과 이미지

Imaage(filename='/content/yolov5/runs/train/pothole_results/train_batch0.png', width=1000)  #train

Imaage(filename='/content/yolov5/runs/train/pothole_results/val_batch0_labels.png', width=1000) #val 검증 -레이블 정보가 뜸

###검증
# !python val.py --weights run/train/pothole_results/weights/best.pt --data ./pothole/data/data.yaml --img 640 --iou 0.65 --half #이미지크기:640

###테스트 test
# !python val.py --weights run/train/pothole_results/weights/best.pt --data ./pothole/data/data.yaml --img 640 --task test

###추론
# %ls runs/train/pothole_results/weights  #best.pt, last.pt
# !python detect.py --weights runs/train/pothole_results/weights/best.pt --img 640 --conf 0.4 --source ./pothole/test/images    #best.pt 로 추론, --source: 테스트 이미지에 대한 추론


