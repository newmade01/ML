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

