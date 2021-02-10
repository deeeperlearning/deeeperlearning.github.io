- 기계 학습 분야에서는 일반적으로 분석적으로 식을 풀어서 답을 얻는 것이 아니라, 조금씩 조금씩 추정치를 바꾸는 작업을 여러 번 반복하여 문제를 해결한다.
- 대다수의 작업은 선형 수식을 풀거나 최소, 최대값을 찾는 최적화 과정인데, 실수(real number)가 포함되는 계산의 경우 디지털 컴퓨터로 정확히 처리하지 못할 수도 있다.

## 4.1 Overflow and Underflow
- 디지털 컴퓨터로 숫자를 처리하는 이상 rounding error가 발생할 수 밖에 없다.
  - underflow: 0에 가까운 숫자가 0으로 처리되어 발생하는 문제
    - 예) 의도치않게 분모에 0이 입력되어 계산 오류가 발생 할 수 있다.
  - overflow: 큰 숫자가 $\pm\infty$로 처리되어 발생하는 문제
    - 예) 안정화되지 않은 값의 원소를 가지는 벡터 x의 softmax 함수 값 계산 <Eq 4.1>
      - x의 원소가 모두 c일 경우: 1/n
      - 하지만 c가 매우 작을 경우: underflow가 발생해 값이 정의되지 않는다.
      - c가 매우 클 경우: overflow가 발생해 값이 정의되지 않는다.
      - 해결 예시) softmax(z) where z = x - max,i(xi)로 만들어 큰 element를 없애 문제를 해결한다.

## 4.2 Poor conditioning
- Conditioning: 입력 변수의 작은 변화에 대해 함수 값이 얼마나 빨리 변하는지
- $A \in \mathbb{R}^{n\times n}$이 eigenvalue decomposition을 가질 때, condition number = <Eq 4.2>
  - 가장 크고 작은 eigenvalue 사이의 비율
  - 이 숫자가 크면 역행렬을 계산하는 등의 작업이 input의 에러에 대해 큰 차이를 만들 수 있음
  - 역행렬을 계산할때의 rounding error 때문이 아니라, 행렬 고유의 성질임
  - Poorly conditioned 행렬은 입력 신호 단계에서 발생한 에러를 증폭 시킴

## 4.3 Gradient-Based optimization
- 대부분의 딥러닝 알고리즘은 f(x)의 최대값이나 최소값에 대한 x를 구하는 최적화 과정을 포함한다.
- f(x): objective function 혹은 criterion, 최소화해야 할때는 cost function, loss function, error function
- f(x)를 최적화해야 할 때 *로 표시 -> x ∗ = arg min f ( x )
- Gradient descent: x 값을 약간 바꾸었을 때 f(x) 값이 감소한다면, 해당 방향으로 x 값을 수정하는 과정을 반복하여 f(x)의 최소값을 만드는 x를 찾는 방법
<fig 4.1>
- f’(x) = 0: critical point, stationary point (local minimum or maximum of saddle point)
<fig 4.2>
- Global minimum: 전체 정의역에 대해 가장 작은 함수값
  - 여러개의 local minima를 갖거나 평평한 saddle point가 많은 경우에는 optimization이 어렵다.
<fig 4.3>
  - critical point: 모든 element가 0
- directional derivative in direction u (a unit vector)
  - 함수 f의 u  방향으로의 기울기
  - 함수 f를 최소화하기 위해 f를 가장 빠르게 감소시키는 방향을 찾고자 함
  - <Eq 4.3, 4.4>
  - u에 무관한 텀을 무시하면, min u cos theta로 간소화되고, u가 gradient와 반대 방향일 때 최소가 됨
  - 위의 방법을 method of steepest descent 혹은 gradient descent라고 함
- 위에서 구한 방향으로 새로운 x값을 찾음
  - <Eq 4.5>
  - $\epsilon$: learning rate
  - Learning rate는 보통 상수를 사용하나, line search라는 방법에서는 몇 가지 learning rate 값을 동시에 테스트해서 f를 최소화 시키는 값을 고름

### 4.3.1 Beyond the Gradient: Jacobian and Hessian Matrices
- Jacobian matrix: f: Rm -> Rn에서 모든 편미분에 대한 행렬
- Second derivative: 미분에 대한 미분으로, curvature를 측정한다고 볼 수 있음
<fig 4.4>
  - -: 아래로 볼록, cost function이 입실론 보다 많이 감소
  - 0: curvature가 없음, cost function의 기울기가 1일 때, 입실론 만큼 감소
  - +: 위로 볼록, cost function이 입실론 보다 적게 감소
- Hessian matrix
  - <Eq 4.6>
  - 다변수를 입력 받을 때의 모든 조합을 고려하는 second derivative와 같은 개념
  - Gradient의 Jacobian이라 할 수 있음
  - 미분의 순서가 바뀌어도 값은 유지되므로, 2차 미분이 continuous하다면 symmetric matrix임
  - Real, symmetric하므로 real eigenvalues와 orthogonal basis eigenvector로 분해 가능
  - $d^{T}Hd$: unit vector $d$방향으로의 2차 미분
  - 2차 미분과 Hessian matrix를 이용하여 적절한 learning rate를 예상 할 수 있음
    - <Eq 4.8 ~ 4.10>
    - 최악의 경우: g가 가장 큰 eigenvalue $\lambda_{max}$와 상응하는 H의 eigenvector의 방향과 일치 -> step size = 1/ $\lambda_{max}$가 됨
- Hessian matrix, H를 이용하면 critical point의 성질을 알 수 있음
  - H가 positive definite (모든 eigenvalue가 양수): local minimum
  - H가 negative definite (모든 eigenvalue가 음수): local maximum
  - 양/음인 eigenvalue가 모두 있음: 방향에 따라 min/max 여부가 다름
<fig 4.5>
- Hessian의 condition number가 클 때, gradient descent는 효과가 좋지 않은 예시
  - 한 방향으로는 급격하게 변하고 (예-$\lambda_{max}$에 대한 eigenvector 방향), 다른 방향으로는 조금 변함 (예-$\lambda_{min}$에 대한 eigenvector 방향)
<fig 4.6>
- Newton’s method로 이와 같은 문제를 해결 가능
  - second-order Taylor series expansion을 이용해 x(0) 근처의 f(x)를 근사함
  - <Eq 4.11, 4.12>
  - 함수 f가 positive deﬁnite quadratic 이면 Newton’s method는 위 식을 이용해서 한번에 minimum으로 갈 수 있음
  - 하지만 현실적으로 f는 국소적으로 positive deﬁnite quadratic이고 전체적으로는 아니므로, 여러번 반복해야 함
  - 이와 같이 gradient descent보다 더 빠르게 critical point로 도달할 수 있지만, local minimum 근처 한정이며 saddle point에서는 오히려 안 좋을 수 있음
- First-order optimization algorithms의 예시 - gradient 만을 이용하는 gradient descent
- Second-order optimization예시 - Hessian matrix를 이용하는 Newton’s method
- 함수에 제약을 걸어 성능을 보장하기도 함: 예) Lipschitz continuous 혹은 Lipschitz continuous derivatives를 가지는 함수
  - 변화율이 Lipschitz constant L에 의해 제한되는 함수 f
  - <Eq 4.13>
  - 입력의 변화가 작을 때, 출력의 변화가 작을 것이라 보장함
- Convex optimization: 강한 제약을 이용해 좋은 성능을 보장함
  - 모든 지점에서 Hessian이 positive semidefinite (eigenvalue가 모두 0 이상)인 confex function에만 적용 가능함
  - 이러한 함수는 saddle point가 없고, local minima가 global minima라 최적화가 용이함

## 4.4 Constrained optimization
- 함수를 최적화 할 때, 모든 x에 대해서 최적화 하는 것이 아니라, 특정 집합 S에 속하는 x에 대해서 최적화 하는 방법
  - S: Feasible points
  - 예시 1: 제약조건을 고려하여 gradient descent를 수정
    - Step size를 정하고 gradient descent step을 만든 후, 결과가 다시 S로 돌아오게끔 projection해줌  
  - 예시 2: Karush-Kuhn-Tucker (KKT)
    - 제약조건이 부등식일 때 적용 가능 (ex-f(x) > 0)
![_config.yml]({{ site.baseurl }}/assets/Ch4_KKT.png)

## 4.5 Example: Linear Least Squares
- 자세한 계산 과정 풀이: https://leejunhyun.github.io/deep%20learning/2018/09/27/DLB-04/
