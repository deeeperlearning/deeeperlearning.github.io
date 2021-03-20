## 8.3 Basic Algorithms

### 8.3.1 Stochastic Gradient Descent (SGD)

- 머신 러닝에서 가장 널리 사용되는 optimization algorithm임

![_config.yml]({{ site.baseurl }}/assets/ch8/Alg8_1.PNG)

- 가장 중요한 parameter: Learning rate, $\epsilon$

- 이전까지는 $\epsilon$가 고정된 경우를 알아봤지만, iteration에 따라 바꿀 수도 있음 ($\epsilon_k$)

- $\epsilon_k$이 고정되어 있다면, local minimum에 도착하고 나서도 noise가 발생할 수 밖에 없음

- SGD의 수렴 조건

  - $\sum_{k=1}^{\infty} \epsilon_k = \infty$
  - $\sum_{k=1}^{\infty} \epsilon_k^2 < \infty$

- $\epsilon$을 trial and error로 정할 수도 있지만, 가장 좋은 방법은 objective function을 시간에 대한 함수로 그려보는 것임

  - $\epsilon$이 너무 높으면, objective function이 빠르게 감소하지만, 강한 oscillation이 발생함 (적당하면 괜찮음)

  - $\epsilon$이 너무 낮으면, learning이 너무 느려서 비용이 높아짐

- SGD의 중요한 성질은, 업데이트 당 연산 시간이 학습 데이터에 수에 비례해서 늘지 않는다는 점임

  - 데이터 양이 매우 많더라도, 전체 학습 데이터를 모두 돌기 전에 특정 범위 이내로 수렴하게 됨


### 8.3.2 Momentum

- SGD가 널리 사용되지만 느리기도 한데, 학습을 가속하기 위해 고안됨

  - 높은 curvature, 작지만 지속적인 gradients, 지저분한 gradients에 효과적임

  - 지난 gradients들을 평균 내어 새로운 방향을 결정함

![_config.yml]({{ site.baseurl }}/assets/ch8/Fig8_5.PNG)

- Parameter 변화의 방향과 속도를 조절하는 $v$ 항을 도입함

![_config.yml]({{ site.baseurl }}/assets/ch8/Eq8_15.PNG)

- Momentum이 도입된 SGD 알고리즘

![_config.yml]({{ site.baseurl }}/assets/ch8/Alg8_2.PNG)

- 예전에는 구간 크기(step size)를 단순히 norm(gradient $\times \epsilon$)으로 사용함

- 하지만 현재는, 최근의 gradients들이 얼마나 크고 일정한지(sequence)에 기반해 결정함

  - 최근의 gradients들이 같은 방향이면 구간 크기가 커짐

  - gradients가 항상 $g$였다면, $-g$ 방향으로 가속함

  - 구간 크기는 $\frac{\epsilon \|\| g\|\|}{1-\alpha}$ 로 설정함

  - 따라서 momentum의 hyperparameter는 $\frac{1}{1-\alpha}$ 이라고 할 수 있음

    - $\alpha=0.9$ 이면 10배로 가속되는 셈으로, $0.5$, $0.9$, $0.99$ 등을 사용

    - $\alpha$도 작은 값에서 시작해 시간에 따라 늘리기도 하지만, $\alpha$를 적절히 정하는 것 보다는 $\epsilon$을 줄이는 것이 더 중요함


### 8.3.3 Nesterov Momentum

- Momentum 알고리즘을 개량하여, estimator 항에 속도항을 반영하는 correction factor를 추가함

![_config.yml]({{ site.baseurl }}/assets/ch8/Eq8_21.PNG)


![_config.yml]({{ site.baseurl }}/assets/ch8/nesterov.jpeg)

- Convex batch gradient case에서는 $k$단계 후 에러의 수렴 속도를 $O(1/k)$에서 $O(1/k^2)$로 줄임

- Stochastic gradient case에서는 수렴 속도에 대한 향상 효과가 없음



## 8.4 Parameter Initialization Strategies

- 사실 initialization 상태에서 학습이 시작되면 어떤 속성이 바뀌는건지 엄밀하게 이해되고 있지 않음

- 따라서 최고라고 할 만한 방법은 없음

  - 아무리 정교하게 정한 방법으로 initialization 하더라도, 학습이 시작되면 weight 분포가 바뀜

- 그나마 한 가지 확실하게 말할 수 있는 initialization의 기능은, 서로 다른 unit들 사이의 대칭성을 깨트린다는 것

  - 예) 만약 두 hidden unit이 같은 input, 같은 activation function을 가진다면 같은 기능 밖에 수행 못함

- 보통 bias는 0(혹은 상수)으로 두고, weight만 randomly initialized함

  - 난수 생성을 위해 보통 임의의 Gaussian 혹은 uniform distribution을 이용하는데, 각 방법의 속성에 대해서는 자세히 연구되지 않음

    - Gaussian 혹은 uniform distribution의 scale을 정하는 여러 가지 방법 소개: https://reniew.github.io/13/
    
  - 큰 weight 값은 강한 대칭성 파괴 효과나 약한 input의 소실을 막지만, recurrent network에서의 chaos 혹은 activation function을 포화시킬 수 있음

  - Regularization에서는 weight를 줄이려 하지만, optimization 측면에서는 초기 weight가 충분히 커야 함


# 8.5 Algorithms with Adaptive Learning Rates

- 학습률은 조절하기 까다로운 초매개변수이다. 학습률에 따라 학습 결과도 많이 달라진다.
- 학습률에 대한 알고리즘 성능의 의존성을 낮추기 위한 운동량 알고리즘등이 있지만 또 다른 초매개변수를 도입해야 하는 문제가 있다.
- 이 절에서는 비용함수의 매개변수 공간에서 스스로 학습 속도를 조절할 수 있는 알고리즘들을 소개한다.

## 8.5.1 AdaGrad

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-adagrad.png)

- 각 매개변수에 대한 기울기의 제곱을 스탭마다 누적한 후 매개변수를 업데이트 할 때 기울기의 역수를 가중치로 할당한다.
- 기울기가 굉장히 가파른 방향으로는 학습 속도가 빠르게 감소하고 기울기가 완만한 방향으로는 학습 속도가 천천히 감소한다. 결과적으로 완만한 경사를 따라 학습이 진행된다.
- Empirical한 결과들을 보면, 훈련 시작부터 유효 학습 속도가 필요 이상으로 빠르게 감소하는 문제가 있다. 이때문에 일부 모델에서는 잘 작동하지만 전반적으로 잘 작동하는 방법은 아니다.
    - 참고로 비용함수가 볼록함수인 경우에 빠르게 수렴하도록 만들어진 알고리즘.

## 8.5.2 RMSProp

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-rmsprop.png)

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-rmsprop-newterov.png)

- RMSProp은 비볼록함수인 비용함수에 대하여 잘 작동하도록 AdaGrad를 수정한 알고리즘.
- AdaGrad에서 기울기 누적하는 부분을 exponentially weighted moving average로 교체한 것.
    - 이렇기 때문에 오래전의 기울기값은 기울기 누적값에 미치는 영향이 작음.
    - Local convexity에 대하여만 AdaGrad를 적용한다고 생각하면 될듯?
- Exponentially weighted moving average의 길이를 결정하는 $\rho$는 초매개변수이다.
- 기본적인 RMSProp 알고리즘과 (Algorithm 8.5) 네스테로프 운동량을 반영한 버전 (Algorithm 8.6) 이 있다.
- 현재 실제로도 많이 사용되는 알고리즘 중 하나.

## 8.5.3 Adam

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-adam.png)

- Adaptive Moments의 줄임말.
- Adam은 RMSProp + 운동량 + (bias correction)이라고 생각하면 된다. 그래서 알고리즘에  1차, 2차 모멘트가 모두 등장한다 (RMSProp은 2차 모멘트만 등장).
- Bias correction이란 1차, 2차 모멘트를 $1-\rho_1^t, 1-\rho_2^t$로 나누어 보정하는 과정을 말한다. 따라서 RMSProp 알고리즘의 초기에 2차 모멘트가 크게 편향될 수 있는 반면 Adam은 그렇지 않다.
- 최근 출판되는 논문들을 보면 Adam이 가장 흔하게 사용되는 것 같다.

## 8.5.4 Choosing the Right Optimization Algorithm

- 그래서 어떤 최적화 알고리즘을 쓰는게 좋은지 물어본다면, 아직 답은 없다.
    - Adaptive learning rate를 이용하는 알고리즘들(RMSProp, AdaDelta, ...)이 전반적으로 좋은 성과를 내긴 했다.

- 여러가지 task에 대하여 여러가지 최적화 알고리즘을 비교한 논문도 있는데, 하나의 알고리즘이 특출나게 좋은 성능을 보이지는 않았다.
# 8.6 Approximate Second-Order Methods

- 이번 절에서는 2차 미분을 사용하는 방법들을 소개한다.
- 이번 절에서는 아래처럼 비용함수에 정칙화 항이 없는 경우에 대해서만 소개하지만, 정칙화 항 등을 추가한 비용함수로도 잘 일반화된다.

$$J(\theta) = \text{E}_{x, y \sim \hat{p}_{data}(x, y)} [L(f(x;\theta), y)] = \frac{1}{m}\sum_{i=1}^m L(f(x^{(i)}, \theta), y^{(i)})$$

## 8.6.1 Newton’s Method

![_config.yml]({{ site.baseurl }}/assets/ch8/newton-method.jpg)

- 현재 매개변수가가 있는 점에서 목적함수를 2차근사 (테일러 전개) 한 후 그사된 2차함수의 최소점으로 매개변수를 업데이트한다.

$$J(\theta) \approx J(\theta_0) + (\theta - \theta_0)^T\nabla_\theta J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^T H\vert_{\theta=\theta_0} (\theta - \theta_0)$$

$$\theta^* = \theta_0 - [H\vert_{\theta=\theta_0}]^{-1} \nabla_\theta J(\theta_0)$$

- Newton's method는 기본적으로 $H$가 양의 정부호 행렬임을 가정한다. 그렇지 않으면 매개변수 업데이트시 실제 극소점에서 멀어질 수 있다. 예를들어 딥러닝의 비용함수에 Netwon's method를 그대로 사용할 경우 saddle point를 만나면 $H$에 양이 아닌 고윳값이 존재하게 되며 해와 멀어지는 방향으로 이동한다. 이러한 문제는 정칙화로 해결할 수 있다.
    - 아래처럼 정칙화를 하게 되면 음의 고유값들을 적당한 $\alpha$를 선택하여 없앨 수 있다.
    - 당연히 음의 고유값이 큰 경우 $\alpha$를 크게 잡으면 $H$의 영향보다 정칙화 항의 영향이 더 커서 최적화에 문제가 생긴다.

$$\theta^* = \theta_0 - [H\vert_{\theta=\theta_0}+\alpha I]^{-1} \nabla_\theta J(\theta_0)$$

- 위와 같이 saddle point 문제도 있지만, 가장 큰 문제는 계산 비용이다. $H$의 역행렬을 계산하기 위한 비용이 굉장히 크기 때문에 신경망같이 매개변수가 많은 상황에서 사용하기는 어렵다.

## 8.6.2 Conjugate Gradients

- Hessian matrix의 이점을 취하면서 계산 비용은 낮춘 최적화 기법이다.

![_config.yml]({{ site.baseurl }}/assets/ch8/converge-newton.jpg)

- 이 알고리즘은 최대 경사법 (책 4.3절) 의 문제를 보완한 알고리즘인데, 최대 경사법의 문제는 다음과 같다.
    - $t_0$에서 line search 방향이 $d_{t_0}$라고 하면, $t_0$가 끝난 후 $d_{t_0}$ 방향으로의 기울기는 0이다.
    - $t_1$에서 $d_{t_0}$방향의 기울기가 0이기 때문에 $d_{t_1}$과 $d_{t_0}$는 직교한다. 이 때문에 위 그림처럼 계단형 궤적이 생긴다.
    - 문제는 $t_0$에서 개선한 것을 $t_1$에서 일정부분 다시 되돌린다.
- 이러한 문제를 해결하기 위해 켤레 기울기법에서는 이전 방향과 켤레인 방향을 선택한다 (아래 식의 $\beta_t$를 잘 선택하여 켤레인 방향을 만들어야 함). 이렇게하면 이전 방향 목적함수의 최솟값을 유지하며 다음 업데이트를 할 수 있다.

$$d_t = \nabla_\theta J(\theta) + \beta_t d_{t-1}$$

- 문제는 $d_t^T H d_{t-1}$의 고윳값을 찾아서 $\beta_t$를 찾아야 한다는 점이다. 이렇게 할 경우 계산 비용 면에서 뉴턴법과 별 차이가 없다. 다행히도 $H$를 계산하지 않고도 $\beta_t$를 계산할 수 있는 방법들이 있다.
    - Flather-Reeves: $\beta_t = \frac{\nabla_\theta J(\theta_t)^T\nabla_\theta J(\theta_t)}{\nabla_\theta J(\theta_{t-1})^T\nabla_\theta J(\theta_{t-1})}$
    - Polak-Ribiere: $\beta_t = \frac{(\nabla_\theta J(\theta_t)-\nabla_\theta J(\theta_{t-1}))^T\nabla_\theta J(\theta_t)}{\nabla_\theta J(\theta_{t-1})^T\nabla_\theta J(\theta_{t-1})}$

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-conjugategrad.jpg)

- 이 방법은 목적함수가 2차함수임을 가정한다. 딥러닝 모델의 비용함수는 2차함수가 아니므로 위의 방법을 그대로 적용하긴 어렵다. 단, 조금 변형하여 비선형 켤레 기울기법을 사용할 수 있다.
    - 딥러닝 목적함수인 경우 켤레 기울기법을 사용해도 이전 방향 목적함수의 최솟값을 보장하지 못한다. 이를 보완하기 위해 비선형 켤레 기울기법은 가끔씩 변경되지 않은 기울기를 따라 line search를 다시 시작하는 방법으로 작동한다.
    - 실제 적용 사례를 보면 딥러닝 모델에서 어느정도 작동한다. 하지만, 이거 한번 계산할 시간에 경사하강법 여러번 계산하는게 효율이 좋다고 한다.

## 8.6.3 BFGS

- BFGS (Broyden-Fletcher-Goldferb-Shanno) 역시 켤레 기울기법 처럼 뉴턴법의 장점은 취하면서 계산 비용은 낮춘 알고리즘이다.
- BFGS는 켤레 기울기법에 비해 뉴턴법의 업데이트 식을 더 직접적으로 근사한다.
- BFGS에서는 반복적인 low rank 업데이트를 통해 $H$의 역행렬을 근사한다.
- 이렇게 $H^{-1}$을 근사한 $M_t$를 이용하여 하강 방향을 정하고 ($\rho_t = M_tg_t$) line search를 하여 하강 크기 $\epsilon^*$를 결정한 후 매개변수를 업데이트한다.

$$\theta_{t+1} = \theta_{t} + \epsilon^* \rho_t$$

- 이렇게 계산 비용은 줄일 수 있지만 큰 크기의 $M$을 계속 저장해야 한다는 문제점이 있다. 즉, $O(n^2)$의 메모리가 필요하기 때문에 딥러닝 모델들에서는 실용적이지 못하다.
- BFGS의 큰 메모리 사용량을 줄이기 위한 방법들도 있다. L-BFGS (limited memory BFGS) 는 이전 스탭의 $M_{t-1}$를 저장하는 대신 이전 스탭의 $M_{t-1}$이 단위행렬이라고 가정하고 $M_t$를 계산한다.
