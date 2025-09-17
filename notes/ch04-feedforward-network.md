
# ch4. Feedforward-network 강의자료 예습

---

## 1. **Feedforward Networks (순전파 신경망, 전방향 신경망)**

* 딥러닝에서 가장 기본적이고 전형적인 모델을 말합니다.
* 다른 이름: **Deep Feedforward Networks**, **다층 퍼셉트론(MLP, Multi Layer Perceptron)**.
* 많은 딥러닝 모델(예: CNN, RNN 등)의 기반이 되는 중요한 개념이에요.

---

## 2. **이 모델의 목표**

* 어떤 **함수 $f^*$** 를 근사(approximate)하는 것.

* 예: **분류기(classifier)**

  * 입력 $x$ → 어떤 범주(category) $y$ 로 매핑
  * 즉, "이 사진이 고양이인지 개인지" 같은 문제.

* 수학적으로는 이렇게 표현해요:

<img width="1820" height="379" alt="image" src="https://github.com/user-attachments/assets/e96807ab-c779-4598-a7a3-08153fc363f7" />


  여기서

  * $x$: 입력 데이터 (예: 이미지, 문장 등)
  * $y$: 출력 (예: "고양이", "개" 같은 레이블)
  * $\theta$: 모델이 학습하는 **매개변수(가중치, weight)**
  * 목표는 "가장 좋은 $\theta$"를 학습하는 것.

---

## 3. **왜 ‘Feedforward(순전파)’라고 부를까?**

* 입력에서 출력으로 정보가 **한 방향으로만** 흘러가기 때문이에요.
* 중간에 출력이 다시 입력으로 들어가는 **피드백 루프가 없음**.
* **순환 신경망(Recurrent Neural Networks, RNN)** 은 출력이 다시 입력으로 들어가서 차이가 있어요.

---

## 4. **왜 ‘Networks(네트워크)’라고 부를까?**

* 하나의 함수가 아니라, **여러 개의 함수들을 합성**해서 만든 모델이기 때문이에요.
* 이 함수들의 연결 구조를 **그래프(특히 DAG, Directed Acyclic Graph)** 로 표현할 수 있어요.

예:

$$
f(x) = f_3(f_2(f_1(x)))
$$

* $f_1$: 첫 번째 레이어(은닉층)
* $f_2$: 두 번째 레이어
* $f_3$: 세 번째 레이어
* 이렇게 함수가 연속적으로 연결된 구조를 **체인(chain)** 이라고 해요.
* 이 체인의 길이(몇 번 합성했는지)가 바로 모델의 **깊이(depth)** → 여기서 **딥러닝(Deep Learning)** 이라는 말이 나온 거예요.

---

# 1. **Preliminary (기초 개념)**

### (1) **실수값 함수 (Real-valued function)**

* 수학적으로는:

<img width="883" height="348" alt="image" src="https://github.com/user-attachments/assets/f284f82e-2232-4219-b05d-661374777cd5" />


  → **입력**: $n$-차원 벡터 $x = (x_1, x_2, \dots, x_n)$
  → **출력**: 실수값 하나 $y$
* 예: 집 크기(㎡), 방 개수, 위치 정보($n$개의 변수) → 집값(실수 하나)

---

### (2) **벡터 함수 (Vector function)**

* 수학적으로는:

<img width="1160" height="332" alt="image" src="https://github.com/user-attachments/assets/6ca9bdcd-348b-4f1a-800f-26bb6048c2a4" />


  → **입력**: $n$-차원 벡터 $x$
  → **출력**: $m$-차원 벡터 $y = (y_1, y_2, \dots, y_m)$
* 예: 한 이미지를 입력하면, (고양이 확률, 개 확률, 토끼 확률) 같은 여러 개의 값이 동시에 나오는 경우.

---

# 2. **뉴런(Neuron)**

* 하나의 뉴런은 **실수값 함수**예요.
* 식:

<img width="1048" height="388" alt="image" src="https://github.com/user-attachments/assets/0561a7bf-ac20-4951-926f-ecc45dfba7cb" />


  * $x = (x_1, x_2, \dots, x_n)$: 입력 벡터
  * $w = (w_1, w_2, \dots, w_n)$: 가중치 벡터
  * $b$: 편향(bias)
  * $activation$: 비선형 함수 (예: ReLU, sigmoid, tanh)
* 즉, **입력에 가중치를 곱하고(b), 합친 뒤, 활성화 함수를 적용한 것**이 뉴런의 출력.

---

# 3. **레이어(Layer)**

* 여러 뉴런이 모이면 하나의 **레이어(층)** 가 됨.
* 식:

<img width="1702" height="451" alt="image" src="https://github.com/user-attachments/assets/d0ae2f94-fafc-4821-9387-53057320acc4" />


  * $W$: 가중치 행렬 (여러 뉴런을 한 번에 표현)
  * $b$: 편향 벡터
  * 출력은 여러 개 ($y_1, y_2, \dots, y_m$)
* 즉, **레이어 = 벡터 함수** (입력 벡터 → 출력 벡터).

---

# 4. **신경망 구조 (Architecture & Notation)**

* 신경망은 보통 이렇게 층으로 구성돼요:

  * **입력층(input layer)**: $x_1, x_2, \dots$
  * **은닉층(hidden layer)**: 중간 계산 담당
  * **출력층(output layer)**: 최종 예측 결과

* 각 층에서 일어나는 연산:

<img width="1585" height="761" alt="image" src="https://github.com/user-attachments/assets/00e71b67-a434-44c9-a6dd-973c37c48817" />

  * $a_j^{[l]}$: $l$-번째 층의 $j$-번째 노드(뉴런)의 출력
  * $w_{jk}^{[l]}$: 가중치
  * $b_j^{[l]}$: 편향
  * $g^{[l]}$: 활성화 함수

* **파라미터(θ)** = 가중치 $W$, 편향 $b$

* 이 값들을 학습(training) 과정에서 조정함.

---

# 5. **학습(Training)**

* 목표: 모델 $f(x)$ 가 우리가 원하는 실제 함수 $f^*(x)$ 와 비슷해지도록 만드는 것.
* **훈련 데이터**: 입력 $x^{(i)}$ 와 원하는 출력 $y^{(i)} \approx f^*(x^{(i)})$
* 중요한 점:

  * 출력층의 목표값은 직접 알려주지만,
  * 은닉층(hidden layer)의 출력은 훈련 데이터가 직접 알려주지 않음.
  * 그래서 **학습 알고리즘(=역전파, 경사하강법)** 이 은닉층 가중치를 자동으로 조정해야 함.

---

# 6. **선형 모델의 한계와 극복 방법**

* **선형 모델 예시**: 선형회귀, 로지스틱 회귀

  * 장점: 효율적이고 해석 용이, 수학적으로 풀기 쉬움 (convex optimization)
  * 단점: **복잡한 패턴(XOR 같은 비선형 관계)을 표현할 수 없음**

* **해결 방법**:

  * 입력 $x$를 그대로 쓰는 대신, 변환된 입력 $\phi(x)$를 사용
  * 즉, 선형 모델을 “비선형 표현”에 적용하는 것.

<img width="1571" height="946" alt="image" src="https://github.com/user-attachments/assets/4e173479-c33f-4731-967f-c128df59f578" />


* 신경망에서는 이 $\phi(x)$ 를 **은닉층과 활성화 함수**가 자동으로 학습해서 만들어줌.

---

# 1. **How to overcome limitation of linear models (선형 모델의 한계 극복 방법)**

선형 모델(로지스틱 회귀, 선형 회귀 등)은 단순해서 강력하지만, **비선형 패턴**(예: XOR 문제)을 표현할 수 없어요. 이를 극복하려면 입력 $x$ 를 변환된 표현 $\phi(x)$ 로 바꿔줘야 합니다.

### (1) $\phi$를 고르는 방법

* **매우 일반적인 $\phi$ 사용**

  * 예: RBF 커널처럼 무한 차원 표현
  * 장점: 훈련 데이터는 항상 맞출 수 있음 (capacity ↑)
  * 단점: 일반화 성능이 떨어짐 (테스트 데이터 성능 ↓ → 과적합)

* **수작업으로 $\phi$ 설계 (feature engineering)**

  * 전통적 머신러닝에서 많이 사용 (텍스트 → 단어 개수, 이미지 → 색 히스토그램 등)
  * 장점: 도메인 지식 활용 가능
  * 단점: 사람 손이 많이 필요하고, 다른 문제에 그대로 옮겨 쓸 수 없음

* **$\phi$를 학습**

  * 신경망 접근 방식
  * 모델이 스스로 $\phi(x)$ (즉, 특성 표현)를 학습하도록 함
  * 구조:

<img width="1165" height="238" alt="image" src="https://github.com/user-attachments/assets/7e82cafe-9c6a-47cb-97d8-8731bd98c984" />


  * $\theta$: 표현 $\phi$를 학습하는 파라미터 (은닉층 가중치)
  * $w$: 최종 출력으로 연결하는 가중치

즉, 딥러닝의 핵심은 **좋은 표현 $\phi(x)$ 를 직접 배우는 것**이에요.

---

# 2. **Design decisions for feedforward networks (모델 설계 시 고려사항)**

### (1) 선형 모델과 공통된 부분

* 옵티마이저 선택 (SGD, Adam 등)
* 비용 함수 선택 (MSE, cross-entropy 등)
* 입력/출력 유닛 형태 정의 (예: 입력은 벡터, 출력은 확률 등)

### (2) Feedforward network 고유의 부분

* **은닉층(hidden layers)** 몇 개?
* **각 층의 뉴런 수**는 몇 개?
* **활성화 함수(activation function)** 무엇을 쓸까? (ReLU, sigmoid, tanh 등)
* **아키텍처 구조** 어떻게 연결할까? (완전연결, 합성곱, 순환 등)

---

# 3. **Training a Neural Network (신경망 학습)**

* 신경망은 비선형 구조 → **비용 함수가 비볼록(non-convex)**

  * 선형 모델(볼록 최적화)과 달리, 전역 최적해(global optimum) 보장은 없음

* 보통 **경사 기반 최적화(gradient-based optimization)** 사용

  * 대표적: 확률적 경사하강법 (SGD)

* 단점:

  * 수렴 보장 없음
  * 초기값(가중치, 편향)에 매우 민감

    * **가중치**: 작은 랜덤 값으로 초기화
    * **편향**: 0 또는 작은 양수로 초기화

* **그래디언트 계산**은 복잡하지만, **역전파(Backpropagation)** 알고리즘으로 효율적으로 수행 가능

---

# 4. **Cost Functions (비용 함수)**

* 비용 함수 = **기본 손실(loss) + 정규화 항(regularization term)**
* 딥러닝에서 가장 많이 쓰는 접근: **최대우도추정(Maximum Likelihood, MLE)**

  * 음의 로그 가능도(Negative Log-Likelihood, NLL)
  * → 분류 문제에서는 **교차 엔트로피(Cross-Entropy) Loss**

### 예시

<img width="1515" height="521" alt="image" src="https://github.com/user-attachments/assets/81f88e9d-2722-48ff-a52f-ce0419781b92" />


   * MSE(평균제곱오차), MAE(평균절대오차)도 쓸 수 있지만,
   * **신경망에서는 gradient가 잘 안 흐르는 문제(gradient vanishing)** 때문에 잘 안 쓰이고,
   * 대신 **Cross-Entropy**가 더 일반적임.

---

# 5. **Likelihood Function (가능도 함수)**

* **가능도(likelihood)**: 주어진 모델 파라미터 $\theta$ 에 대해 관측 데이터 $X$ 가 나올 확률

<img width="1386" height="103" alt="image" src="https://github.com/user-attachments/assets/1c10a55c-6877-477e-86fd-cc8745e41fa4" />


* 즉, “이 파라미터라면 지금 가진 데이터가 나올 확률은 얼마인가?”

* 예: 주사위

  * 공정한 주사위라면 각 눈이 나올 확률 = $1/6$
  * 어떤 데이터 $X=(4,3,3)$ 가 나왔다면,

    * 공정한 주사위일 확률
    * 편향된 주사위일 확률
  * 비교해서 어떤 모델이 더 그럴듯한지 판단 → 이게 **우도추정(MLE)**

---

# 1. **Maximum Likelihood Estimation (최대우도추정, MLE)**

* **아이디어**: 어떤 확률 분포(모델)를 가정했을 때, 주어진 데이터가 가장 그럴듯하게 나올 수 있도록 파라미터 $\theta$ 를 선택하자.
* 수학적 정의:

<img width="1004" height="120" alt="image" src="https://github.com/user-attachments/assets/a6367e80-5995-41c6-aa96-6be2c91dae15" />


  여기서

  * $\mathcal{L}(\theta|X)$: 우도 함수 (데이터 $X$가 주어졌을 때 파라미터 $\theta$일 확률)
  * $\arg\max$: 우도를 최대로 만드는 $\theta$ 값 선택

---

### 🎲 예제: 주사위

* **공정한 주사위**: 각 눈 확률 = $1/6$
* **편향된 주사위**: 예를 들어 $(1/8, 1/8, 3/8, 1/8, 1/8, 1/8)$
* 데이터 $X_1 = (4,3,3)$, $X_2 = (5,2,5)$ 가 나왔다고 하자.

→ 어떤 주사위가 더 가능성이 큰가?

* MLE는 바로 이런 식으로, **데이터에 가장 적합한 모델 파라미터를 찾는 방법**이에요.
* 선형 회귀(Linear Regression)에서 **최소제곱법(least squares)** 은 사실상 MLE의 특별한 경우임.

---

# 2. **Logistic Regression: Optimization (로지스틱 회귀 최적화)**

### (1) 모델 정의

* 입력: $d$-차원 벡터 $x \in \mathbb{R}^d$
* 파라미터: 가중치 $w = (w_0, w_1, \dots, w_d)$
* 확률적 모델:

  $$
  p(y=1|x; w) = \sigma(w^T x) \quad \text{(시그모이드 함수)}
  $$

---

### (2) 우도 함수 (Likelihood)

* 데이터셋: $(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)$

* **독립이고 동일하게 분포(IID)** 가정 → 전체 우도는 곱으로 표현 가능:

  $$
  L(w) = \prod_{i=1}^N p(y_i | x_i; w)
  $$

* 이진 분류일 때, 개별 항:

  $$
  p(y_i|x_i; w) = \big(p(y=1|x_i; w)\big)^{y_i} \cdot \big(1 - p(y=1|x_i; w)\big)^{(1-y_i)}
  $$

  → $y_i = 1$ 이면 첫 항만 남고, $y_i = 0$ 이면 두 번째 항만 남음.

---

### (3) 로그 우도 (Log Likelihood)

* 계산 편하게 로그 취함:

  $$
  \log L(w) = \sum_{i=1}^N \Big( y_i \log p(y=1|x_i; w) + (1-y_i)\log(1 - p(y=1|x_i; w)) \Big)
  $$

---

### (4) 최적화 문제

* 우리는 **우도 최대화** 대신 **음의 로그 우도(NLL, Negative Log-Likelihood)** 최소화를 자주 씀:

  $$
  -\log L(w) = - \sum_{i=1}^N \Big( y_i \log p(y=1|x_i; w) + (1-y_i)\log(1 - p(y=1|x_i; w)) \Big)
  $$
* 이게 바로 로지스틱 회귀의 **손실 함수(loss function)**, 흔히 **이진 크로스 엔트로피(Binary Cross-Entropy)** 라고 불러요.

---


## 📌 Output Units (출력층의 역할)

* 출력층은 \*\*숨겨진 표현(hidden features)\*\*을 최종적으로 \*\*예측값(ŷ)\*\*으로 변환하는 단계예요.
* 예:

  * 입력 → 은닉층 → **출력층 → ŷ**

출력층에서 어떤 함수를 쓰느냐에 따라 모델이 표현할 수 있는 확률 분포나 값이 달라집니다.

---

## 1. **Linear Units (선형 출력 유닛) → 회귀 문제**

* 식:


<img width="1566" height="124" alt="image" src="https://github.com/user-attachments/assets/36dcf66d-a05a-4100-b250-be1940348836" />


* **용도**: 연속적인 값(실수) 예측 (예: 집값, 온도, 주가 등).
* 해석:

  * 출력값 = 평균(μ) 역할을 하는 값.
  * 분포는 \*\*가우시안(정규분포)\*\*이라고 가정:


<img width="1246" height="87" alt="image" src="https://github.com/user-attachments/assets/f3dcd112-2017-4866-adcf-60bd33c8ceb8" />

    
* 학습 목표: **로그우도 최대화 ↔ MSE 최소화**
  → 즉, 회귀 문제에서는 자연스럽게 **Mean Squared Error**가 코스트 함수로 연결됨.

---

## 2. **Sigmoid Units (시그모이드 출력 유닛) → 이진 분류**

* 식:


<img width="654" height="224" alt="image" src="https://github.com/user-attachments/assets/3086ca77-01ba-4526-b191-9962b07adf5e" />

  
* 출력 범위: (0, 1) → **확률**처럼 해석 가능.
* **용도**:

  * $y \in \{0,1\}$인 **이진 분류(binary classification)** 문제.
  * 예: 환자에게 병이 있는지(Yes/No), 이메일이 스팸인지(Yes/No).
* 특징:

  * sigmoid 덕분에 항상 0\~1 사이 값이 나와서 확률로 해석 가능.
  * 코스트 함수는 **이진 cross-entropy** 사용.

---

## 3. **Softmax Units (소프트맥스 출력 유닛) → 다중 분류**

* 식:

<img width="1184" height="197" alt="image" src="https://github.com/user-attachments/assets/a0ef14b3-f77b-471a-98e5-44d760de4e4a" />

  
* 출력: $K$개의 클래스에 대한 확률 분포 (합 = 1).
* **용도**: 다중 클래스 분류 (예: 개/고양이/새).
* 특징:

  * sigmoid의 다중 클래스 버전.
  * 각 클래스 확률을 한 번에 표현 가능.
  * 코스트 함수는 **다중 클래스 cross-entropy**.

---


## 📌 Hidden Units (은닉 유닛)

* 은닉 유닛은 입력을 받아서 **비선형 변환**을 해주는 기본 구성요소.
* 수식:

<img width="1146" height="252" alt="image" src="https://github.com/user-attachments/assets/7cbee534-f193-4bbe-83ea-c666d074dc2c" />

  
* 여기서 \*\*활성화 함수 g(z)\*\*의 선택이 은닉 유닛을 구분하는 기준.

---

## 📌 Activation Functions (활성화 함수)

은닉층에서 어떤 비선형성을 넣어주느냐에 따라 네트워크의 표현력이 달라집니다.

<img width="1447" height="770" alt="image" src="https://github.com/user-attachments/assets/cc93e821-5de8-408c-88e6-079dec694001" />



### 1. **ReLU (Rectified Linear Unit)**

$$
g(z) = \max(0, z)
$$

* 장점:

  * **양수 영역에서는 선형** → 기울기 소실(gradient vanishing) 문제 없음.
  * 계산 간단, 빠른 학습.
* 단점:

  * 음수 영역에서는 gradient = 0 → **죽은 뉴런(dead neuron)** 문제 발생.

---

### 2. **ReLU 일반화**

<img width="1548" height="938" alt="image" src="https://github.com/user-attachments/assets/72378ecf-415a-4d3a-bb97-1b81c948ef2f" />


---

### 3. **Maxout**

* z 벡터를 k개 그룹으로 나누고, 각 그룹의 최대값을 선택.
* 장점: **활성화 함수를 직접 학습** 가능 → ReLU나 다른 함수도 근사 가능.
* 단점: 파라미터 수가 크게 늘어남.

---

### 4. **Sigmoid & Tanh**

<img width="1316" height="448" alt="image" src="https://github.com/user-attachments/assets/9b5de9e8-3c46-4673-8533-9b2f840c46f1" />


  (사실상 sigmoid를 변형한 버전)
* 단점:

  * 입력이 크거나 작으면 gradient → 0 (기울기 소실, vanishing gradient).
  * exp 계산으로 비효율적.
* 특징:

  * **은닉층에서는 잘 안 씀**,
  * 대신 출력층(binary classification)에서는 sigmoid,
    출력층(multi-class)에서는 softmax를 여전히 사용.
  * 꼭 써야 한다면 **tanh가 sigmoid보다 나음** (0 중심, 더 학습 잘 됨).

---

## 📌 Architecture Design (네트워크 설계)

* **구성 요소**:

  * 층 수(depth)
  * 층당 유닛 수(width)
  * 연결 방식

* 원칙:

  * 깊어질수록(더 많은 레이어) 복잡한 함수 근사가 가능.
  * 단순히 뉴런만 늘리는 것보다 **깊게 쌓는 게 효과적**.
  * 하지만 너무 깊어지면 학습이 어려워짐(gradient vanishing/exploding).

* **Universal Approximation Theorem**:

  * 충분히 큰 MLP(다층 퍼셉트론)는 **이론적으로 임의의 함수**를 근사 가능.
  * 하지만 학습이 실제로 잘 되리라는 보장은 없음.

* **실무에서 모델 크기 결정 방법**:

  * Grid search, Random search, Neural architecture search 등 **탐색 기반 방법**.

---

