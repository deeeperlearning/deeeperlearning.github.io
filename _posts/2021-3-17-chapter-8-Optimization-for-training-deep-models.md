## 8.3 Basic Algorithms

### 8.3.1 Stochastic Gradient Descent (SGD)

- 머신 러닝에서 가장 널리 사용되는 optimization algorithm임

[Alg8.1.PNG]

- 가장 중요한 parameter: Learning rate, $\epsilon$

- 이전까지는 $\epsilon$가 고정된 경우를 알아봤지만, iteration에 따라 바꿀 수도 있음 ($\epsilon_k$)

- $\epsilon_k$이 고정되어 있다면, local minimum에 도착하고 나서도 noise가 발생할 수 밖에 없음

- SGD의 수렴 조건

  - $\sum_{k=1}^{\infty} \epsilon_k = \infty$
  - $\sum_{k=1}^{\infty} \epsilon_k^2 < \infty$

- $\epsilon$을 trial and error로 정할 수도 있지만, 가장 좋은 방법은 objective function을 시간에 대한 함수로 그려보는 것임

  - $\epsilon$이 너무 높으면, objective function이 빠르게 감소하지만, 강한 oscillation이 발생함 (적당하면 괜찮음)

  - $\epsilon$이 너무 낮으면, learning이 너무 느려서 비용이 높아짐

- SGD의 가장 중요한 성질은, 업데이트 당 연산량이 학습 데이터에 수에 비례하지 않는다는 점임

  - 데이터 양이 매우 많더라도, 수렴할 수 있도록 함


### 8.3.2 Momentum

- SGD가 널리 사용되지만 느리기도 한데, 학습을 가속하기 위해 고안됨

  - 높은 curvature, 작지만 지속적인 gradients, 지저분한 gradients에 효과적임

  - 지난 gradients들을 평균 내어 새로운 방향을 결정함

[Fig8_5.PNG]

- Parameter 변화의 방향과 속도를 조절하는 $v$ 항을 도입함

[Eq8_15.PNG]

- Momentum이 도입된 SGD 알고리즘

[Alg8_2.PNG]

- 예전에는 구간 크기(step size)를 단순히 norm(gradient $\times \epsilon$)으로 사용함

- 하지만 현재는, 최근의 gradients들이 얼마나 크고 일정한지(sequence)에 기반해 결정함

  - 최근의 gradients들이 같은 방향이면 구간 크기가 커짐

  - gradients가 항상 $g$였다면, $-g$ 방향으로 가속함

  - 구간 크기는 $\frac{\epsilon||g||}{1-\alpha}$ 로 설정함

  - 따라서 momentum의 hyperparameter는 $\frac{1}{1-\alpha}$ 이라고 할 수 있음

    - $\alpha=0.9$ 이면 10배로 가속되는 셈으로, $0.5$, $0.9$, $0.99$ 등을 사용

    - $\alpha$도 작은 값에서 시작해 시간에 따라 늘리기도 하지만, $\alpha$를 적절히 정하는 것 보다는 $\epsilon$을 줄이는 것이 더 중요함


### 8.3.3 Nesterov Momentum

- Momentum 알고리즘을 개량하여, estimator 항에 속도항을 반영하는 correction factor를 추가함

[Eq8_21]

- Convex batch gradient case에서는 $k$단계 후 에러의 수렴 속도를 $O(1/k)$에서 $O(1/k^2)$로 줄임

- Stochastic gradient case에서는 수렴 속도에 대한 향상 효과가 없음


![_config.yml]({{ site.baseurl }}/assets/ch7/norm_fig_1.png)
