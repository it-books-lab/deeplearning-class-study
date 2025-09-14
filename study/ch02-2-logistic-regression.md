# logistic regression code 보충 설명

## 모델 커스텀 코드

```python
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

```

- **`torch.nn.Module`**
    - 커스텀 모델을 만들 때 반드시 `nn.Module`을 상속받아야 함.
    - 내부적으로 `forward()`, `backward()` 같은 메커니즘이 구현되어 있어 학습 가능.
- **`__init__`**
    - 모델의 "구성요소"를 정의.
    - 여기서는 `self.linear = nn.Linear(input_dim, output_dim)` 이 핵심
        - `input_dim`: 입력 벡터의 크기 (예: MNIST에서 28×28 픽셀 → 784 차원).
        - `output_dim`: 출력 벡터의 크기 (예: 숫자 0~9 → 10차원).
        - 즉, 784 → 10으로 변환하는 **선형 변환 (W·x + b)** 을 정의한 것.
- **`forward()`**
    - 입력 데이터를 받아 순전파(계산)를 어떻게 할지 정의합니다.
    - `self.linear(x)` → 입력 `x`를 선형 변환해서 10개의 "점수(score)"를 만듭니다.
    - `torch.sigmoid()` → 각 점수를 (0,1) 사이 확률처럼 보이게 바꿔줍니다.
        
        하지만 **실제 다중분류 문제에서는 CrossEntropyLoss 안에 Softmax(활성함수 중 하나)가 이미 포함되어 있기 때문에 sigmoid는 필요 없음.**
        

---

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

- **`CrossEntropyLoss`**
    - 분류(classification)에서 자주 쓰는 손실 함수.
    - 내부적으로 `Softmax` + `Log-Loss` 가 합쳐져 있음.
    - 모델이 만든 10개 점수(로짓, logits)를 `Softmax`로 확률처럼 바꾼 뒤, 정답 레이블과 비교해 손실값을 계산.
    - 예: 정답이 `3`인데, 모델이 `[0.01, 0.05, 0.1, 0.7, 0.14,...]` 라면 손실이 낮음.
- **`optim.SGD`**
    - 확률적 경사 하강법(Stochastic Gradient Descent)
    - `model.parameters()`는 모델 안의 학습 가능한 가중치(W, b)
    - `lr=lr` 은 학습률(learning rate) → 한 번 업데이트할 때 얼마나 크게 움직일지.

---

## Training code

```python
loss_save_arr = []
for i in range(epochs):
    # train
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train.long())

    loss.backward()
    optimizer.step()

    loss_save_arr.append(loss.data)

```

1. **`loss_save_arr = []`**
    - 에포크마다 계산한 손실(loss)을 저장해둬서 나중에 그래프로 그리거나 분석할 수 있습니다.
2. **`for i in range(epochs):`**
    - `epochs`만큼 반복 학습.
    - 예: 10 에포크라면, 전체 훈련 데이터를 10번 학습한다는 뜻.
3. **`model.train()`**
    - 모델을 "훈련 모드"로 전환.
    - Dropout, BatchNorm 같은 레이어는 학습/평가 모드에서 동작이 달라지는데, 여기서는 그냥 **학습 모드**라고 생각하면 됨.
4. **`optimizer.zero_grad()`**
    - 이전 단계에서 계산된 기울기(gradient)를 모두 0으로 초기화.
    - PyTorch는 기본적으로 gradient를 계속 누적하기 때문에, 매번 학습 시작 전에 꼭 초기화해야 함을 주의할 것.
5. **`output = model(X_train)`**
    - 학습 데이터(`X_train`)를 모델에 넣어 순전파(forward) 결과를 얻음.
    - 출력은 (batch_size × output_dim) 모양의 행렬, 즉 각 샘플이 10개 클래스 중 어디에 속할지에 대한 점수 벡터.
6. **`loss = criterion(output, y_train.long())`**
    - 예측값(`output`)과 실제 정답(`y_train`)을 비교해 손실(loss)을 계산.
    - `y_train.long()` 은 CrossEntropyLoss가 레이블을 **정수 인덱스(예: 0~9)** 로 받아야 하기 때문.
7. **`loss.backward()`**
    - 역전파(Backpropagation)를 수행.
    - 즉, 손실값이 각 파라미터(W, b)에 얼마나 영향을 미쳤는지 미분(gradient)을 계산함.
8. **`optimizer.step()`**
    - 계산된 기울기(gradient)를 바탕으로 실제 W, b를 업데이트.
    - 예: `W_new = W_old - lr * dW`
9. **`loss_save_arr.append(loss.data)`**
    - 이번 에포크의 손실값을 저장.
    - 나중에 학습 곡선(Loss vs Epoch) 그릴 때 사용.

---

전체 흐름:

**[입력 → 선형 변환 → 출력 → 손실 계산 → 역전파 → 파라미터 업데이트]**

---

## Test code

### `_, pred = torch.max(output.data, axis=1)`

- **`torch.max(tensor, axis=1)`** 은 2차원 텐서에서 **각 행(row, axis=1)마다 가장 큰 값과 그 인덱스(숫자 0~9)**를 반환.
- 반환값은 `(최댓값, 최댓값의 인덱스)`인데, 여기서 최댓값은 필요 없으니까 `_`로 버리고, 인덱스만 `pred`에 저장.

---

### `float((pred == y_train).sum()) / y_train.size(0)`

- **정확도(accuracy)를 계산**하는 코드.
1. **`pred == y_train`**
    - 예측(pred)과 정답(y_train)을 비교 → 맞으면 `True(=1)`, 틀리면 `False(=0)`.
2. **`.sum()`**
    - 맞은 개수를 센다.
    - 예: 64개 중 50개 맞았으면 `sum() = 50`.
3. **`float(...) / y_train.size(0)`**
    - `y_train.size(0)` = 배치 크기 (예: 64).
    - `맞은 개수 / 전체 개수` → 정확도.
    - 위 예시라면 `50 / 64 ≈ 0.781` (정확도 78.1%).

---

