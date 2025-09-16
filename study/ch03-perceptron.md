# Ch03. Perceptron

## Multi-layer perceptron

- Multi-layer perceptron(MLP) → Non-linear 문제 해결

---

## Artificial neuro

<img width="1583" height="1081" alt="image" src="https://github.com/user-attachments/assets/860dfeef-b314-491c-a511-dcf6efdb2ff6" />

---

## Linear models as a artificial neuron
 : 머신러닝 모델마다 어떤 함수를 쓰고, 어떤 목표를 최소화하는지를 요약

<img width="1753" height="854" alt="image" src="https://github.com/user-attachments/assets/0ee5e6a4-4b4a-4a80-bc4b-173815690ff7" />

- Perceptron 학습은 "예측이 틀린 샘플"이 있을 때만 가중치를 업데이트하여 분류 오류를 줄이는 방향으로 동작.
즉, Perceptron은 "맞냐 틀리냐 자체를 줄이는 것"이 목표.

- mean squared error(MSE) = 오차^2의 평균
즉, Linear Regression은 "실제 숫자와 예측 숫자의 차이를 최소화"하는 게 목표.

<img width="1730" height="591" alt="image" src="https://github.com/user-attachments/assets/a9810d90-3dce-4f5a-abf8-0b73fdb0a29e" />

- Logistic Regression은 "정답 클래스의 확률을 최대한 높이는 방향으로 학습"하는 게 목표.

---

## Perceptron Learning Algorithm (PLA)

<img width="1772" height="1121" alt="image" src="https://github.com/user-attachments/assets/c8007061-5817-4780-a8a8-d41119a578b7" />


---

## Formulation of Linear Regression

<img width="2135" height="1067" alt="image" src="https://github.com/user-attachments/assets/ebec4d2e-6a70-40a9-b4a1-9b03e61821a5" />

- MSE 식을 미분해서 0이 되는 지점을 찾는 방식

---

## Formulation of Logistic Regression

<img width="1417" height="822" alt="image" src="https://github.com/user-attachments/assets/731816d1-d76c-46dd-936c-25fee587ef9e" />


---

### 1️⃣ Cross-Entropy Loss의 수식

Logistic Regression에서 binary classification일 때 **loss**는 이렇게 정의됩니다:

$$
L(y, p) = - \Big( y \cdot \log(p) + (1-y) \cdot \log(1-p) \Big)
$$

* $y$: 실제 정답 (0 또는 1)
* $p$: 모델이 출력한 "1일 확률" (sigmoid 출력, 0\~1 사이 값)

---

### 2️⃣ y=1일 때

정답이 \*\*1(양성 클래스)\*\*라면:

$$
L = - \big( 1 \cdot \log(p) + 0 \cdot \log(1-p) \big) = - \log(p)
$$

* $p$가 1에 가까우면 → $-\log(p)$ ≈ 0 (loss 작음 ✅)
* $p$가 0에 가까우면 → $-\log(p)$ → 무한대 (loss 큼 ❌)

👉 즉, \*\*정답이 1일 때는 "p도 1이어야 한다"\*\*는 걸 강하게 유도.

---

### 3️⃣ y=0일 때

정답이 \*\*0(음성 클래스)\*\*라면:

$$
L = - \big( 0 \cdot \log(p) + 1 \cdot \log(1-p) \big) = - \log(1-p)
$$

* $p$가 0에 가까우면 → $-\log(1-p)$ ≈ 0 (loss 작음 ✅)
* $p$가 1에 가까우면 → $-\log(1-p)$ → 무한대 (loss 큼 ❌)

👉 즉, \*\*정답이 0일 때는 "p도 0이어야 한다"\*\*는 걸 강하게 유도.

---

## Gradient Descent

<img width="1326" height="544" alt="image" src="https://github.com/user-attachments/assets/491e678e-e1fa-4498-89d1-250108781bdd" />

---

<img width="1324" height="626" alt="image" src="https://github.com/user-attachments/assets/94c5d28c-2dbf-467e-b72a-c57de16ec3a0" />


1. **m=1: Stochastic Gradient Descent (SGD)**
    - 데이터 샘플을 **하나씩** 뽑아서 매번 gradient 업데이트.
    - 장점: 빠르고 online 학습 가능.
    - 단점: 노이즈가 크고 수렴이 불안정할 수 있음.
2. **1<m<N: Mini-batch SGD**
    - 데이터 일부(minibatch)를 사용해 gradient 계산.
    - **실무에서 가장 많이 사용**됨.
    - 장점:
        - 벡터 연산(병렬 처리) 덕분에 효율적.
        - 노이즈가 줄어 수렴 안정적.
    - mm 크기는 보통 32, 64, 128 등으로 설정.
3. **m=N: Batch Gradient Descent**
    - 전체 데이터셋으로 gradient 계산.
    - 장점: gradient가 정확.
    - 단점: **매 iteration 비용이 너무 큼** (특히 데이터가 크면 비효율적).
    - 실무에서는 거의 안 쓰이고, 이론적인 기준점으로 주로 언급됨.

---

## Backprogation(역전파)

참고자료 - https://www.youtube.com/watch?v=1Q_etC_GHHk

### forward pass

<img width="1620" height="631" alt="image" src="https://github.com/user-attachments/assets/31d75b32-7825-4eca-af83-85271c7ae683" />

### backward pass와 chain rules

- 졸림이슈로 타이핑 대신.. 손필기 사진으로..

<img width="3024" height="2608" alt="image" src="https://github.com/user-attachments/assets/bbdbe530-5e2c-4a7c-877a-59bbac96b002" />
















