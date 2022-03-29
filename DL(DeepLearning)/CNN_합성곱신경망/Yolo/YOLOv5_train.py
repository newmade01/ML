#데이터셋정리
from glob import glob

train_img_list = glob('content/yolov5/pothole/train/images/*.jpg')
test_img_list = glob('content/yolov5/pothole/test/images/*.jpg')
valid_img_list = glob('content/yolov5/pothole/valid/images/*.jpg')

# text파일로 이미지 리스트 파일 저장
import yaml

with open('content/yolov5/pothole/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('content/yolov5/pothole/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')

with open('content/yolov5/pothole/valid.txt', 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

