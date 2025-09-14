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


