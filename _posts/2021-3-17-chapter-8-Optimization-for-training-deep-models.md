## 8.3 Basic Algorithms

### 8.3.1 Stochastic Gradient Descent (SGD)

- 머신 러닝에서 가장 널리 사용되는 optimization algorithm임

[Alg8.1.PNG]

- 가장 중요한 parameter: Learning rate, $\epsilon$

- 이전까지는 $\epsilon$가 고정된 경우를 알아봤지만, iteration에 따라 바꿀 수도 있음 ($\epsilon_k$)

- $\epsilon_k$이 고정되어 있다면, local minimum에 도착하고 나서도 noise가 발생할 수 밖에 없음

- SGD의 수렴 조건

  - $\sum_{k=1}^{\infty} \epsilon_k = \infty$
  - $\sum_{k=1}^{\infty} (\epsilon_k)^2 < \infty$



![_config.yml]({{ site.baseurl }}/assets/ch7/norm_fig_1.png)
