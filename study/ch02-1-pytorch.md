```python

model = LogisticRegression(max_iter=10, tol=0.0001, penalty='l2', C=1.0)

model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
train_acc = metrics.accuracy_score(y_train, y_train_predict)
print(train_acc)
y_test_predict = model.predict(X_test)
test_acc = metrics.accuracy_score(y_test, y_test_predict)
print(test_acc)
```

- `max_iter=10`: **최대 10번 반복하면서 가중치 수정**
- `tol=0.0001`: 수렴 조건, 손실 변화량이 이 값보다 작으면 중간에 학습 종료


---


```python
x = torch.ones([2,3,1])
print(x)

###실행 결과###
tensor([[[1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.]]])
```

---

- 일반적으로는 numpy에서 tensor로 옮기는 경우가 많음. AI 학습 돌리기 전에 데이터 전처리 시각화 등을 먼저 해보기때문에!

```python
import numpy as np
import torch

data = [[1,2,3], [4,5,6]]

x = torch.tensor(data)
print(x)
data_arr = np.array([[4,3],[1,3],[5,5]])
x = torch.tensor(data_arr)
print(x)
x = torch.from_numpy(data_arr)
print(x)
x = torch.randn([3,5]) #표준 정규분포(평균 0, 표준편차 1)에서 샘플링한 랜덤 값으로 텐서를 생성
print(x)
```

---

```python
print(tensor[0]) # 0번째 행
print(tensor[:,0]) # 0번째 열
print(tensor[...,-1]) # 마지막 열
tensor[:, 1] = 0
print(tensor)

###실행 결과###
tensor([ 1.2993, -0.1316, -0.2037])
tensor([ 1.2993, -0.1050])
tensor([-0.2037,  2.2178])
tensor([[ 1.2993,  0.0000, -0.2037],
        [-0.1050,  0.0000,  2.2178]])
```

- `…, -1`: 차원이 몇 개 있든(…) 상관없이, 마지막 축의 -1번째 요소 선택


---


```python
y = torch.tensor([2], dtype=torch.float64, requires_grad=True)
z = x * y
z.backward()
print(x.grad) # dy/dx = y = 2
print(y.grad) # dx/dy = x = 5
```

- PyTorch에서 **자동 미분(autograd)** 을 활성화할지 결정하는 옵션입니다.
- `requires_grad=True`로 설정하면,
    - 해당 텐서에서 일어나는 모든 연산을 **계산 그래프에 기록**합니다.
    - 나중에 `.backward()`를 호출하면, 이 텐서에 대한 **기울기(gradient)** 가 자동으로 계산됩니다.


---


```python
# 원소별 곱
y1 = torch.mul(tensor_one, tensor_two)
y2 = tensor_one*tensor_two

# matrix multiplication
y1 = torch.matmul(tensor_one, tensor_two.T)
y2 = tensor_one@tensor_two.T
```

---


```python
# tensor → numpy / numpy → tensor를 했을 때
# tensor와 numpy array가 cpu에 있다면 메모리를 공유함

t = torch.ones(5)
print(f"t: {t}")

n = t.numpy()
print(n)

t.add_(3)
print(t)
print(n)

###실행 결과###
t: tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
tensor([4., 4., 4., 4., 4.])
[4. 4. 4. 4. 4.]

```

---

```python
# squeeze 텐서 차원 축소
x1 = torch.randn(3,1,1,3)
print(x, x1.shape)
x2 = x1.squeeze() # 크기 1인 차원을 제거
print(x2, x2.shape)
print()

# unsqueeze 텐서 차원 확장
x1 = torch.randn(3,1,3)
print(x, x1.shape)
x2 = x1.unsqueeze(dim=0) # 특정 위치(dim)에 크기 1인 차원을 추가
print(x2, x2.shape)
```



