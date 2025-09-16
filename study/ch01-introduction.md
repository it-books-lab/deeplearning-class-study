# Introduction 보충 공부

## 1. AI - ML - Representation Learning - Deep Learning

### 1. **AI (Artificial Intelligence, 인공지능)**

- **가장 큰 범위**.
- 사람이 하는 지능적 활동(문제 해결, 학습, 추론, 의사결정 등)을 기계가 할 수 있게 만드는 기술 전반을 뜻함.
- 예: 체스 두기, 언어 번역, 자율주행.

---

### 2. **ML (Machine Learning, 기계학습)**

- AI의 한 분야.
- 기계가 **명시적으로 프로그래밍하지 않아도(if문을 달아서 적정 온도를 유지하는 에어컨 프로그램)**, 데이터에서 패턴을 학습해서 스스로 성능을 개선하는 기술(알아서 사람 수, 날씨에 따라 적정 온도를 설정하는 로직을 만드는 것).
- 예: 스팸 메일 분류기 (메일 데이터를 보고 "스팸"과 "정상"의 패턴을 학습).

---

### 3. **Representation Learning (표현 학습)**

- ML 안에서도 더 구체적인 개념.
- 데이터에서 **유용한 특징(Feature, Representation)을 자동으로 추출**해서 학습에 활용하는 방법.
- 예: 이미지에서 "고양이 귀 모양", "눈동자 위치" 같은 특징을 사람이 직접 설계하지 않고, 모델이 스스로 찾아내는 것.

---

### 4. **Deep Learning (딥러닝)**

- Representation Learning의 한 종류이자, ML의 하위 분야.
- **신경망(Neural Network)을 깊게 쌓은 구조**를 이용해서 데이터의 복잡한 패턴을 학습.
- 이미지, 음성, 자연어 같은 **비정형 데이터 처리에 특히 강력함**.
- 예: ChatGPT, 이미지 생성 모델, 음성 인식.

---

- 인공지능 – 기계가 인간의 지능을 모방할 수 있도록 하는 기술이나 규칙 기반의 응용 프로그램.
- 머신러닝 – 데이터를 통해 학습할 수 있도록 수학과 통계를 통합한 인공지능의 하위 집합.
- 표현학습 – 원시 데이터에서 피처 탐지 또는 분류를 위해 필요한 표현을 자동으로 발견할 수 있는 머신러닝의 하위 집합.
- 딥러닝 – 구조화되지 않거나 레이블이 지정되지 않은 데이터에서 학습하기 위해 신경망을 사용하는 표현학습의 하위 집합.

---

## 2. Target function, Hypothesis

<img width="1440" height="763" alt="image" src="https://github.com/user-attachments/assets/46fa8821-c3f4-447c-85b3-b207459c6373" />

<img width="1251" height="873" alt="image" src="https://github.com/user-attachments/assets/724c4203-4ffa-47c0-8a71-11f1380b736c" />


### 1. Target function (목표 함수, 이상적인 함수)

- **정의**: 우리가 실제로 찾고 싶은, "진짜 세상의 규칙"을 말합니다.
- 즉, 고객의 신용카드 발급 승인 여부를 **완벽하게 판별할 수 있는 이상적인 규칙**이에요.
- 하지만 현실에서는 이 함수 **f** 가 뭔지 정확히 알 수 없고, 단지 데이터(과거 기록)를 통해 **추측**할 수 있을 뿐입니다.

> 예시: "연봉이 일정 금액 이상이고, 현재 빚이 연봉의 절반 미만이면 승인" 같은 규칙이 실제 은행 내부에 존재할 수 있지만, 우리는 그걸 모른다고 가정합니다. 이게 Target function이에요.
> 

---

### 2. Hypothesis (가설 함수, 모델)

- **정의**: 우리가 데이터와 학습 알고리즘을 통해 **추정하려는 규칙**입니다.
- 보통 신경망(Neural Network), 선형 회귀(Linear Model) 같은 **모델 구조** 안에서 학습을 통해 얻어진 함수 g 가 Hypothesis에 해당합니다.
- 즉, Target function을 **근사**하려는 "우리의 베스트 시도"입니다.

> 예시: 신경망을 학습시켰더니 나온 함수가 “0.7×연봉 – 0.5×부채 + … > 0 이면 승인, 아니면 거절” 같은 규칙을 만들었다면, 이게 Hypothesis입니다.
> 

---

## 3. perceptron, perceptron learning algorithm(PLA)

<img width="1772" height="974" alt="image" src="https://github.com/user-attachments/assets/59e685d5-7254-42ca-9172-52da8f70dea4" />

<img width="1954" height="950" alt="image" src="https://github.com/user-attachments/assets/9ac74ff9-c6b1-4c12-b076-c743b5237ae6" />

---

 ## 4. Supervised/Unsupervised Learning
 
 - Supervised - (binary/multi-class)classification, regression 둘 다 가능!
 - Unsupervised - clustering, dimension reduction 가능!
 - 따라서, 아까 '2. Target function, Hypothesis'에서 본 예제처럼
 - input, output이 주어진 supervised learning에서
 - regression과 classification 중 뭘 쓰면 좋을 지 판단하면 됨(즉, regression과 classification이 후보 Hypothesis들이 되는 것!)

 ---

 ## 5. Reinforcement Learning

- 누적 보상 개념을 도입하여 intelligent agents를 학습시키는 것

 ---

 ## 6. Supervised Learning 흐름

 <img width="1030" height="481" alt="image" src="https://github.com/user-attachments/assets/eab9a65c-9cf3-40f5-8ae1-281f7e83a5f2" />

- 궁극적으로 파라미터인 세타를 구해서 '2. Target function, Hypothesis' 그림에서의 최종 Hypothesis를 결정하는 것이 목표임!
- 어떻게 세타를 구할 건가? 

 <img width="1725" height="330" alt="image" src="https://github.com/user-attachments/assets/02682393-ba25-44cf-af37-5be87920d444" />

- cost function을 계산하여 최대한 h(x)가 y에 가깝게 하는 파라미터를 구한다.
- 얼마나 가깝게 만들 것인가? -> 오버핏하지 않는 선에서!

---

## 7. Unsupervised Learning Example - K-means clustering

- **목적**: 데이터들을 비슷한 것끼리 **K개의 그룹(cluster)** 으로 나누는 것.
- **입력**: 몇 개의 군집으로 나눌지(`K`)를 사용자가 미리 정해줘야 함.
- **출력**: 각 데이터가 속하는 클러스터와, 각 클러스터의 중심점(centroid).

1. **초기화**
    - 데이터를 K개의 무리로 나눌 건데, 먼저 랜덤하게 K개의 중심점(centroid)을 선택함.
2. **할당 단계 (Assignment step)**
    - 각 데이터가 **가장 가까운 중심점**에 속하도록 클러스터에 할당.
    - 거리 계산은 보통 **유클리드 거리** 사용.
3. **업데이트 단계 (Update step)**
    - 각 클러스터에 속한 데이터들의 **평균값**으로 중심점을 다시 계산.
4. **반복**
    - 중심점이 더 이상 크게 움직이지 않거나, 일정 횟수 반복 후 종료.


---

## Perceptron 보충 설명
퍼셉트론은 1957년 프랭크 로젠블라트가 고안한 인공 신경망 모델로, 여러 입력값을 받아 특정 임계값(threshold)을 넘으면 1을, 그렇지 않으면 0(또는 -1)을 출력하는 알고리즘입니다. 이는 인간의 뉴런을 모방한 최초의 인공지능 모델이며, 입력 데이터의 가중치를 조정하여 학습하고 선형 분류가 가능한 알고리즘이지만, XOR 문제와 같은 비선형적인 문제는 해결할 수 없는 한계


