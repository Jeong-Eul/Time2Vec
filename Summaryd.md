# Time2Vec: Learning a Vector Representation of Time    

<blockquote>
<br>
Author: Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota,<br>Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker<br>
<br>
Date: 11 Jul 2019<br>
</blockquote>

<br>

## Abstract  

Time을 모델링하는 것은 매우 중요하다. 최근 연구에서 새로운 아키텍처를 도입하여 이 문제를 다루고 있는데, 본 논문에서는 그러한 논문과 다른 접근법을 사용하고(Orthogonal) 기존에 존재하는 여러 모델들과 쉽게 결합하여 성능을 높일 수 있는 <b>학습 가능한</b> 시간 임베딩 모델을 제안했다.  

## Introduction  


논문에서 언급하는 <b>시간 변수(Time feature)</b>가 중요한 이유: <br>
<br>
시간 변수를 이용하는 여러 예제들

1. 시간의 흐름에 따라 기록된 환자의 medical history를 기반으로 next health event 예측  
2. 시간의 흐름에 따라 기록된 고객의 listening history를 기반으로 노래 추천  
3. etc.. 

<br>
시간을 포함한 문제(problems involving time)를 풀기 위한 모델의 입력 값은 데이터에 i.i.d(시행이 서로 독립이고, 동일한 분포를 따르는 것)<br> 한 시퀀스 이기보다, data points와 상호적으로 존재하는 시퀀스이다.(서로 관련이 있다)
<br>
<br>

- daily sales의 경우 높은 판매량을 달성했을 때, 그 날이 Holiday여서 그럴 수 있으며<br>
- 환자의 재방문 시점 예측의 경우 이전 방문들의 비주기적 패턴과 다소 관련이 있을 수 있다.<br>  

<br>

<i>(Q1)</i> 그럼 ICU에서 각각의 medical concept이 수집된 시간과 서로 상호관계가 있을까?(<b>단, 환자가 달라도 말이다.</b>)
<br>

<blockquote> 즉, 이런 가설을 세울 수 있다.<br>chartevent에 수집된 시간 기록(offset)은 환자가 바뀌든 아니든 상관 없이 수집된 시간의 간격이라든가 scale이 서로 비슷하고,<br> 이는 labtest에 수집된 시간 기록의 특징과는 다르다.</blockquote>

<br>

논문에서 <b>제안하고자 하는 것</b>:<br>  
<p align = 'center'><img src ="https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/introduction.jpg?raw=true" width = 80%></p>

 - 학습 가능한 representation (learnable vector represntation)  
 - 다양한 모델에 쉽게 적용 가능 (easily combined with many models or architectures)

<br>

## Related Work  

<blockquote>
<br>
<p align ='center'><b>Algoritms for predictive modeling in Time series analysis</b></p><br>
</blockquote>
<br>

1) Auto-Regressive: 미래의 값을 예측하기 위해서 특정 윈도우 내의 과거(자기 자신)의 값을 활용하는 방식   
$\to$ 윈도우의 크기를 얼마나 길게 잡아야하는지 명확하지 않음

2) Hidden Markov models, Dynamic Bayesian networks, conditional random fields: Time step 별 hidden state를 도입하여 과거의 정보를 미래에 반영하는 방식 $\to$ RNN과 비슷하지만 이들은 input sequence에 대한 가정이 현실적이지 않음.  

많은 선행 연구에서 시계열 관련 모델을 제안했지만 이 논문의 목표는 새로운 시계열 분석 모델을 제안하는 것이 아니라, 대신 다양한 모델에서 사용될 수 있는 시간의 벡터 임베딩 형태인 "Time2Vec"을 제안하는 것이다.  

벡터 임베딩을하는 방법은 이전에 ,text(bow 등), graph(그래프 임베딩을 말하는 듯)등 다른 도메인에서도 성공적으로 사용되었다. Time2Vec은 시간 신호를 일련의 frequency로 인코딩하는 시간 분해 기술과 관련이 있으면서도, Fourier 변환과 달리 아예 frequency를 학습할 수 있도록 설계되었다.

다른 논문들은 시간을 고려한 새로운 신경망 구조를 제안하는데, 이 논문은 그런게 아니라 하나의 아키텍처에서 Time2Vec을 활용하여 시간 정보를 더 잘 활용하는 방법을 제안하고, 실제로 실험에서 LSTM의 다양한 변형구조에 time2vec을 활용해서 성능을 높였다.

## Time2Vec   

논문에서 저자는 Time2Vec을 설계하면서, 다음의 3가지 특성을 가질 수 있도록 설계하고자 했다.<br>

1) <b>Periodicity</b>: Time2Vec의 결과로 나온 representation vector가 periodic or non-periodic 한 패턴을 모두 Capture할 수 있어야 한다.<br>
$\to$ 몇 가지 disease들은 나이가 많을 수록 발병할 수 있다.(non-periodic)


2) <b>Invariance to Time Rescaling</b>: 어떠한 resolution을 갖고 있는 Time으로 Time2Vec을 사용했을 때의 결과가 time rescaling을 한 후 데이터를 입력해도 같은 특성을 파악할 수 있어야 함<br>
$\to$ day 단위로 학습된 Time2Vec이 hour 단위의 데이터를 받았을 때도 같은 특징을 representation 할 수 있어야 함(1일 = 24시간)  

3) <b>Simplicity</b>: 어떠한 모델에도 입력될 수 있도록 매우 간단해야함  
$\to$ matrix representation 같은 경우 다른 input과 결합되기 어려움  



이 모든 조건을 충족하는 Time2Vec의 수식은 다음과 같다.  

<p align='center'><img src="https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/time2vec.jpg?raw=true" width=80%></p>

>> notation 정리<br>
>>> $\tau$ : scalar notion of time  
>>> $i$ : element number(time sequence 중 몇 번째 요소인지?)  
>>> $w_{i}$, $\varphi_{i}$: Learnable parameter  
>>> $k+1$ : Vector size  
>>> $F$ : Sinusoid Function(cosine 함수를 사용해도 무방함)  
>>> <b>t2v</b>$(\tau)$ : vector representation  



- sin 함수의 특성에 따라 t2v의 주기는 $\frac{2\pi }{w_{i}}$ 가 되며, t2v $(\tau)$의 값은 t2v $(\tau + \frac{2\pi }{w_{i}})$의 값과 같다.
- $w_{i}$는 주기를 학습하는 파라미터이고, $\varphi_{i}$는 Phase-shift를 학습한다.
<br>
<p align='center'><img src="https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/phaseabc.jpg?raw=true"></p>

- time scale에 강건한 이유는 $w_{i}$를 데이터를 기반으로 학습할 수 있기 때문이다.  
    실험에서 ($\tau$가 day 단위일 때) 7일을 주기로 labeling 되어 있는 시간데이터로 학습된 <b>t2v</b>$(\tau)$의 $w_{i}$는 $\frac{2\pi }{7}$에 근사되었고 <br> 
    $\to$ 이 데이터를 그대로 가지고 day를 2배 했을 때 14일 주기로 맞출 수 있으면 되는데 실제로 그랬음 ($\frac{2\pi }{2\times7}$)<br>
    $\to$ 즉, $\frac{2\pi }{7} \times \tau = \frac{2\pi }{2 \times 7} \times (2\times\tau)$

    - 데이터 예시(출처: <a href="https://github.com/ojus1/Time2Vec-PyTorch/tree/master">Time2Vec Github repo)</a>
    <p align='center'><img src ="https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/toy.jpg?raw=true"></p>

- sin 함수 사용으로 unseen data의 time을 외삽(extrapolating)할 수도 있다.(어디다 쓰지?)  
- 사실 이런 아이디어는 Transformer의 positional encoding에서 착안했음을 밝히고 있다.  
- 같은 word여도, 서로 다른 position에 존재할 경우 다른 의미를 갖을 수 있는데, 시간도 마찬가지이다.  
- Time2Vec은 특정 Time을 단순히 vector로 만드는 것이 아니라, 전체 시점을 모두 고려해서 주기적, 비주기적 패턴<br>(phase-shift을 모델링$\to$ positional encoding과 다른점)을 찾을 수 있다.  


## Experiments & Result  

이 논문은 신기하게 5가지 질문을 정의하고, 이 질문에 대한 답을 찾기위해 ablation study를 진행했다.  

><br><b><i>Question1.</i></b>: is Time2Vec a good representation for time?<br>  
<b><i>Question2.</i></b>: can Time2Vec be used in other architectures and improve their performance?<br>  
<b><i>Question3.</i></b>: what do the sine functions learn?<br>  
<b><i>Question4.</i></b>: can we obtain similar results using non-periodic activation functions for Eq.(23) instead of periodic ones?<br>  
<b><i>Question5.</i></b>: is there value in learning the sine frequencies or can they be fixed?(fourier or exponentially-decaying values as in Vaswani et al.[57]'s positional encoding)<br>  
<br>


### Dataset in figure  

1. Event-MNIST: MNIST 데이터를 Flat 한 후 픽셀 값이 0.9 보다 큰 위치를 기록한 데이터이다. Event-MNIST는 각각의 픽셀이 시간에 따라 변하는 동적인 데이터로 변환되는데, 이는 이벤트 카메라라고 불리는 카메라를 사용하여 빛의 변화나 움직임이 감지되는 순간에 데이터를 기록할 수 있다.  

2. N_TIDIGITS18: 오디오 데이터셋으로, 시간 t와 주파수 채널 c의 시퀀스 집합으로 구성된 데이터이다. (t,c)  사람이 0(zero, oh) 부터 9까지 총 11개의 숫자를 말하는데 이것이 기록되어 있다. 이 데이터셋의 task는 어떤 숫자를 말하는 지 맞추는 것이다.

3. Stack Overflow: stack overflow 유저가 시간의 흐름에 따라 받은 badge의 시퀀스 집합으로 구성된 데이터이다. (b, t) 여기서의 task는 미래의 t에 어떤 badge를 받을 것인지를 맞추는 것이다.  

4. LastFM: LastFM 유저의 시간의 흐름에 따른 listening habit이 기록된 데이터셋이다. (song, time) 이 데이터셋의 task는 미래의 시간 t에서 어떤 노래를 들을 것인지를 맞추는 것이다.  

5. CiteULike: 유저가 citeulike website에 포스팅한 시간과 어떤 주제를 포스팅 했는지가 포함되어 있으며 Task는 LastFM과 비슷하다.  

<b>데이터 셋에 대해서 완전히 이해하지는 못했지만, 시간 정보를 잘 모델링할 수 있어야 어떤 Task든 잘 수행할 수 있을 것 같다.</b>


### Model architecture used in experiment 

<b>LSTM-T</b>: 기존 LSTM 모델에 Time 정보를 단순히 concat(또는 feature engineering 후)하여 모델의 입력으로 사용한 모델이다.  

<b>TimeLSTM</b>: 기존 LSTM에 존재하지 않았던 time gate를 추가하여 LSTM을 재구성한 것이다. LSTM의 변형 모델 중 하나로 Gers & Schmidhuber(2000)가 소개한 엿보기 구멍(peephole connection)을 LSTM에 추가한 것인데, gate layer들이 cell gate를 반영하게 만든 모델이다.  
 - TLSTM1, TLSTM2, TLSTM3으로 총 3가지의 변형구조가 존재한다.(본 논문의 Appendix C에서 자세한 내용을 확인할 수 있으며, 'What to Do Next: Modeling User Behaviors by Time-LSTM'이라는 논문에서 증명을 확인할 수 있다.)  

 <p align ='center'><img src = "https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/TLSTM.jpg?raw=true"></p>


<b>LSTM+Time2Vec</b>: LSTM-T에서 단순히 time 정보를 concat 했다면, 이 모델은 Time2Vec의 결과를 concat하여 모델의 입력으로 사용한 모델이다.  

<b>TLSTM(n) + Time2Vec</b>: TimeLSTM의 3가지 구조에서 등장하는 time gate의 입력값으로 Time2Vec을 사용한 모델이다.  

<p align='center'><img src="https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/tlstm_t2v.jpg?raw=true"></p>


### Ablation study 1.: On the effectiveness of Time2Vec: Question1, 2

<p align='center'><img src = "https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/figure1.jpg?raw=true"></p>
<br>

figure에서 x 축은 epoch, y 축은 accuracy와 recall(top 5)를 의미한다. 실험을 통해 알 수있는 사실은 다음과 같다.  

1. 다양한 데이터셋에서 Time2Vec을 활용한 LSTM 모델이, 그렇지 않은 모델(시간 정보를 단순히 사용한)보다 성능이 좋았다.
2. LSTM을 통해 time feature를 추출하는 것 보다, time을 vector로 representation 하여 활용함으로써, LSTM을 잘 optimize 할 수 있다.  
3. (b), (c)같은 long term sequence 데이터의 경우 Time2Vec을 활용했을 때 더 효율적이다.  

위 3가지 사실로부터, <b><i>Question1.</i></b>: is Time2Vec a good representation for time?에 대한 질문을 해결할 수 있다.  

<p align='center'><img src = "https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/figure2.jpg?raw=true"></p>
<br>

위 그림은 TLSTM 같은 기존에 존재하는 모델에 time 대신 t2v를 활용하여 실험한 것을 표현하고 있다. 즉, 다른 모델에 쉽게 결합될 수 있는지 확인하고 싶었던 것 같다. 위 실험을 통해 알 수 있는 사실은 다음과 같다.  

1. Time feature를 사용한 TLSTM1,3 보다 t2v를 사용한 모델의 성능이더 좋았다.  
2. 왜 LSTM에만 t2v를 결합하여 실험했는 지 모르겠다. 내가 연구자였다고 가정했을 때, 진짜로 Simplicity를 증명하고 싶었다면 Bert의 positional encoding 대신에 T2V를 입력하여 성능이 더 잘나오는지 확인했을 것 같다.  


### Ablation study 2.: On the effectiveness of Time2Vec: Question3  

이 실험에서는 T2V의 sin 함수가 어떤 것을 학습할 수 있는지 즉, 데이터와 Task에 맞는 적절한 주기 $w$와 phase-shift $\varphi$를 잘 학습할 수 있는지를 보기위해 진행했음을 알 수 있다.  

실험에서 사용한 데이터셋은 "Synthesized data"로, 단순히 day(일)와 7일 단위로 labeling 된 y가 존재한다. 따라서 본 데이터셋의 Task는 입력된 시간이 7의 배수인지 아닌지 그 여부를 맞추는 문제가 되겠다.  

실험을 위해서 Time2Vec 위에 Fully connected layer를 두고, sigmoid 함수를 통해 클래스에 해당할 확률을 도출할 수 있도록 아키텍처를 구성했다.  

처음에는 input(시간)이 1개씩 들어가서 sigmoid를 취하고 0.5보다 크면 1, 아니면 0으로, 7의 배수인지 아닌지를 맞추는 줄 알았는데, 이렇게 되면 전체 시간의 맥락을 고려할 수 없다.  

누가 이 논문은 깃허브에 코드로 구현해놓은 것이있어 참고해서 이해한 바로는, 일단 batch 단위가 들어가는 것 같다.  
그 후 여기서는 FC의 출력 노드 수를 2개로 한 후에, Crossentropy(softmax 포함)를 통과하여 확률 값을 도출한다.


<b>학습 과정</b>:

1. 데이터셋 불러오기  
$\to$ data.py 에서 볼 수 있다.  

```python
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ToyDataset(Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
        
        df = pd.read_csv("./data/toy_dataset.csv")
        self.x = df["x"].values
        self.y = df["y"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

if __name__ == "__main__":
    dataset = ToyDataset()
    print(dataset[6])
```

toy_dataset.csv 라는 것은 위에서 첨부했던 데이터 예시 사진과 같은 데이터셋이다.  

2. Time2Vec 구성하기  
$\to$ periodic_activation.py에서 확인할수 있다. 여기서 sin 함수를 쓸 것인지, cos 함수를 쓸 것인지 결정할 수 있도록 구현했다.
 

```python
import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
```

주목할 점은 SineActivation에서 i가 0일때 행렬곱을 취할 w, b와 i가 1과 k 사이일때(k-1개) 행렬곱을 취할 w,b를 구분하여 주었다는 것이다. 즉, 주기를 학습할 파라미터와 phase-shift를 학습할 파라미터를 따로 구성한다.  

그 후, t2v함수에서 i=0 일때 sin 함수를 통과하지 않은 v2, $0<i<k+1$  일때 sin 함수를 통과한 v1을 각각 계산한 후에 concat한다. 

이러면 t2v를 통과하고 난 후 데이터의 차원은 batch x outfeature 이다.  

3. Fully connected layer 구성하기  

```python
from periodic_activations import SineActivation, CosineActivation
from Data import ToyDataset
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Model, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim) 
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)
        
        self.fc1 = nn.Linear(hiddem_dim, 2)
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x
```

정의했던 sineActivation을 불러와주고, 모델을 구성한다. sineActivation을 통과하게 되면, 차원이 batch x outfeature 이므로 fc layer의 입력 차원은 outfeature와 동일하며 출력 차원은 2로 구성하여 최종 모델의 출력을 batch x 2로 만들어준다. 

이러면 각 관측치 마다 노드가 2개 나오며 여기서 softmax를 취해서 시간이 7의 배수인지 아닌지 판단할 수 있다.  


4. 훈련 루프 구성하기  


```python
class ToyPipeline(AbstractPipelineClass):
    def __init__(self, model):
        self.model = model
    
    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        dataset = ToyDataset()
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        num_epochs = 100

        for ep in range(num_epochs):
            for x, y in dataloader:
                optimizer.zero_grad()

                y_pred = self.model(x.unsqueeze(1).float())
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()
                
                print("epoch: {}, loss:{}".format(ep, loss.item()))
    
    def preprocess(self, x):
        return x
    
    def decorate_output(self, x):
        return x


pipe = ToyPipeline(Model("sin", 42))

```

맨 아랫부분 객체생성을 보면 인자로 sin 함수를 사용할 것이며, outfeature의 수는 42로 설정했다. batchsize가 2048이므로
아래와 같은 과정으로 차원을 계산할 수 있다.

input(2048,1) -> sinActivation weight((1, 1), (1, 41)) -> t2v(1, 41) , t2v(2047, 41) -> concat(2048, 42) -> FC weight(42, 2) -> output(2048, 2)

---


<p align ='center'><img src = "https://github.com/Jeong-Eul/Time2Vec/blob/main/Image/figure3.jpg?raw=true"></p>

dsadasdsaddsaDS