Time2Vec 논문을 파이썬 코드로 구현해보는 과정입니다.  

사용한 데이터셋은 eICU의 infusionDrug.csv 이며, 여기서 infusion offset은 ICU 입원 후 몇 분이 지났는지를 의미합니다.  

처음 시도했던 것은, 이 테이블의 고유한 time 정보(수집된 간격에 대한 패턴 등)을 학습하고자 독립변수는 offset, 종속변수는 Vasoactive/inotropic 약물 투여 여부였습니다.  
vasoactive/inotropic 약물은 다음과 같습니다.  
'Norepinephrine (mcg/min)', 'Dopamine (mcg/min)', 'Phenylephrine (mcg/min)', 'Epinephrine (mcg/min)', 'Nitroprusside (mcg/kg/min)', 'Dobutamine (mcg/kg/min)'  

하지만 시간 X와 약물 투여 어부는 직관적으로 생각했을 때, 상관관계가 다소 적기 때문에 학습이 잘 되지 않았습니다.  
CEHRT 논문에서 왜 time embedding 정보를 medical concept embedding 정보와 concat하여 모델링 했는지 여기서 이유를 찾을 수 있습니다.  

따라서 subtask를 구성하기 위해서는 medical concept embedding 같은 추가적인 변수 도입이 필요해보입니다.  

그렇기에 이번 폴더에 들어있는 실험은 단순히 Time2Vec이 정말 잘 작동하는지에 대한 실험입니다.
X는 offset이며, y는 7분 단위로 labeling이 되어 있습니다.  

trainset과 testset은 환자가 겹치지 않도록 독립적으로 분리되었으며, class ratio 또한 두 데이터셋에 동일하게 분포하도록 sampling을 진행했습니다.  

hidden node의 수는 42개이며, output node는 기존 구현 버전과 다르게 1개로 고정하고, 마지막 layer에 sigmoid 함수를 씌웠습니다.
loss는 crossentropy, epoch는 150회, learning rate는 10^(-4)입니다.   

자세한 사항은 experyment.ipynb에 있습니다.  


![image](https://github.com/Jeong-Eul/Time2Vec/assets/122766824/9b82a36d-209f-4cff-bd0a-f30e8a0669fe)
