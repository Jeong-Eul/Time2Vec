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

<img width="25" alt="star1" src="https://user-images.githubusercontent.com/78655692/151471925-e5f35751-d4b9-416b-b41d-a059267a09e3.png">

<span style="font-size:120%">(Q1) 그럼 ICU에서 각각의 medical concept이 수집된 시간과 서로 상호관계가 있을까?(<b>단, 환자가 달라도 말이다.</b>)</span>
<br>

<blockquote> 즉, 이런 가설을 세울 수 있다.<br>chartevent에 수집된 시간 기록(offset)은 환자가 바뀌든 아니든 상관 없이 수집된 시간의 간격이라든가 scale이 서로 비슷하고,<br> 이는 labtest에 수집된 시간 기록의 특징과는 다르다.</blockquote>

<br>

즉, 이런 가설을 세울 수 있다.<br>chartevent에 수집된 시간 기록(offset)은 환자가 바뀌든 아니든 상관 없이 수집된 시간의 간격이라든가 scale이 서로 비슷하고,<br> 이는 labtest에 수집된 시간 기록의 특징과는 다르다.
{: .notice--danger}


[Success Button](#){: .btn .btn--success}

