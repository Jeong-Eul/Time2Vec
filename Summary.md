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
- Time2Vec은 특정 Time을 단순히 vector로 만드는 것이 아니라, 전체 시점을 모두 고려해서 주기적, 비주기적 패턴을 찾을 수 있다.


