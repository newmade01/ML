# GAN (Generative Adversarial Network)

### generative model
- 생성모델은 실존하지 않지만 **있을 법한 이미**습를 생성
- 적절한 분포를 학습
- 통계적인 모델로 표현 (확률분포: 확률값이 높은 이미지로 표현하면 이미지가 표현이 됨) + 약간의 노이즈를 섞고, 랜덤한 샘플링 필요
- 이미지 모델 분포를 근사하게 하는 새로운 모델을 만듬
- 학습 시간이 지나며, 생성 모델이 원본 모델에 따라감
- 통계적으로 평균적인 특징을 가진 DATA가 만들어짐
- (ex. 코의 길이, 눈의 길이는 평균적인 수치를 가짐 but, 눈썹의 두께만 다르게)

### GAN
- 생성자(generate) , 판별자(discriminator) 두개의 네트워크 활용 생성모델 (판별자: 생성자를 잘 학습 할 수 있도록 도와줌)
- 노이즈를 생성자에 넣어 가짜 이미지를 만들어
- 생성자: V 값을 낮추려고함 , 새로운 이미
- 판별자: V값을 높이고자함 , 얼마나 진짜 같은지에 대한 확률값 (진짜 이미지: 1 , 가짜 이미지: 0)
- 생성자는 가짜 이미지가 판별자에 의해 진짜 이미지가 될수 있도,=1가 되도록 학습
- 생성자의 값이 판별자에 의해 Loss 값에 의해 업데이트됨
- 판별자는 가짜, 진짜 이미지를 동시에 받아 1 or 0을 부여
- min(G), max(D) 게임
- D와 G을 한번씩 반복해서 학습 (optimal)
- 기댓값 계산: 모든 데이터를 하나씩 확인하여 식에 대입한뒤, 평균 계산 / 각각의 사건 값에 대한 확률값 곱하여 더함

### GAN 수렴과정
1. 생성자의 분포(z: 노이즈값)
2. 원본 학습자의 분포 D(G(Z))
3. 판별자(D)는 최종 가짜, 진짜 이미지를 구분할 수 없 50%값을 가짐 
4. 약간의 노이즈를 섞에서 원본 데이터와 비슷한것에서 새로운 데이터를 만들어냄

### 증명(Global Optimality)
- 생성자의 분포는 원본 Distribution(확률분포)을 따라가게 된다.
- 생성자가 원본 data와 동일할 때, Global Optimum Point를 갖게됨

### 라이브러리 
```python
import torch
import torch.nn as nn #모델정의
from torchvision import tensorflow_datasets
import torchvision.tranasforms as transforms #전처리
from torchvision.utils import save_image
```

### 생성자 , 판별자 모델 정의 
```python
latent_dim = 100 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #하나의 블록 정의
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(inuptut_dim, output_dim)]
            if normalize:
                #배치 정규화 수행(차원동일)
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLu(0.2 , inplace=True))
            return layers
        #생성자 모델은 연속적인 여러개 블록
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1*28*28),
            nn.Tanh() #-1 ~ 1 사이의 값 가짐
        )
    def forward(self, z):
        img = self.model(z) #noize벡터 = z
        img = img.view(img.size(0), 1, 28, 28) #batch_size, 채널, 높이, 너비
        return img

#판별자 클래스 정의
class Discriminatro(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init()

        self.model =nn.Sequential(
            nn.Linear(1*28*28, 512),
            nn.LeakyReLu(0.2, inplace=True)
            nn.Linear(512, 256),
            nn.LeakyReLu(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        #이미지에 대한 판별 결과 반환
        def forward(self, img):
            flattened = img.view(img.size(0), -1)   #하나의 벡터형태로 나열
            output = self.model(flattened)  #모델을 넣음
            
            return output
```

### 학습 데이터셋 불러오기
```python
transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

train_dataset = dataset_MNIST(root='./dataset', train=True, download=True, transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, suffle=True, num_workses=4)

```

###모델 학습 및 샘플링 
```python
#생성자 판별자 초기화
generator = Generator()
discriminator = Discriminator()

generator.cuda() #GPU
discriminator.cuda()

#손실함수
adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

#학습률(learning rate) 설정
lr = 0.0002

#생성자 판별자ㅏ 최적화
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), betas(0.5, 0.999))
```

###  
```python
import time

n_epochs = 200
sample_interval = 2000 #배치 몇번마다 출력
start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        #진짜, 가짜 이미지 정답 레이블 
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) #1
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0)

        real_imgs = imgs.cuda()

        #1.생성자 학습
        optimizer_G.zero_grad()
        #랜덤노이즈 샘플링
        z = torch.normal(mean=0, std=1, size=(img.shape[0], latent_dim)).cuda()
        #이미지 생성
        generated_imgs = generator(z)
        #생성자의 손실값 계싼: generator가 real 이미지가 되도록 계산
        g_loss =  adversarial_liss(discriminator(generated_imgs), real) #
        #생성자 업데이트
        g_loss.backward()
        optimizer_G.stetp()

        #2. 판별자 학습
        optimizer_D.zero_grad()
        #판별자의 손실값 계산
        real_loss = adversarial_loss(discriminator(real_imgs), rea;)
        fake_loss = adversarial_loss(discriminator(generated_imgs.derach()), fake)
        d_loss(real_loss + fake_loss) / 2
        
        #판별자 업데이트
        d_loss.backward()
        optimizer_D.step()
        
        done = epoch.len(dataloader)  + i
        if done % sample_interval ==0:
            save_imge(generated_imgs.data[:24], f"{done}.png", nrow=5, normalize=True)
            
    #하나의 epoch마다 로그
    print(f"[Epoch {epoch}/{n_epochs}] [D ldss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s] ")

```

### 이미지 출력
```python
from Ipython.display import Image
Image('~.png')
```


#GAN 논문
- 새로운 framwork 제안
- 적대적인 두개의 네트워크 모델(생성자, 판별자)로 학습
- G: 원본 데이터의 분포 학습
- D: 한장의 이미지가 학습 데이터인지 or G로 부터 만들어진 데이터인지 구별할 수 있도록 학습
- G & D => minmax게임 (G는 속이려함, D는 속지 않으려함)
- 오직 뉴럴 네트워크로 학습
- 깊은 네트워크 학습
- G: 위조지폐를 만든는 위조범 vs. D: 감별하는 경찰
- dropout , relu 다양한 layer 사용 가능
- **noise Variable(z) => G => fake,real => D => Update**
- 미분가능한 딥러닝 모델
- 결론: 생성자의 분포는 원본 데이터 분포에 수렴
- 임계점이 여러개 존재
- 