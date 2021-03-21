학습 알고리즘에서 사용하는 최적화 방법과 다른 최적화 알고리즘의 차이, 신경망을 최적화하는데 어려움을 주는 현실적인 문제들, 최적화를 위한 몇가지 알고리즘, 적응형 학습률 또는 비용함수의 2차 미분을 사용하는 최적화 알고리즘 등을 다룬다.

## 8.1 How Learning Differs from Pure Optimization

대부분의 머신러닝은 training set을 통해 학습한 후 test set을 통해 performance measure(P)를 측정한다. 즉, training set으로부터 계산된 비용함수를 줄이면 test set에 대한 비용함수도 줄어들 것이라는 기대만으로 최적화를 진행하기 때문에 순수한 최적화와는 차이가 있다.

### 8.1.1 Empirical Risk Minimization

실제 데이터를 생성하는 확률분포를 $p_{data}(\boldsymbol x,y)$라 하면, 최적화의 최종 목적은 아래의 목적함수를 최소화 하는것이다.

$$J^*(\boldsymbol \theta) = \mathbb E_{(\boldsymbol x,y)\sim p_{data}}L(f(\boldsymbol x;\boldsymbol \theta),y)$$

그러나 $p_{data}(\boldsymbol x,y)$를 알 수 없기 때문에 training set의 확률분포에 대해서 최적화를 하는데, 이를 empirical risk minimization이라고 한다.

$$\mathbb E_{(\boldsymbol x,y)\sim \hat p_{data}}L(f(\boldsymbol x;\boldsymbol \theta),y) = \frac{1}{m}\sum^{m}_{i=1}L(f(\boldsymbol x^{(i)};\boldsymbol \theta),y^{(i)})$$

- 오버피팅이 생기기 쉽다.
- 대부분의 알고리즘은 경사하강법을 사용하는데, 0-1 loss 같은 유용한 손실함수는 기울기를 측정할 수 없다.

위의 두가지 이유로 딥러닝에서는 empirical risk minimization을 잘 사용하지 않는다고 한다.

### 8.1.2 Surrogate Loss Functions and Early Stopping

손실함수를 직접 사용하지 않고, 대리함수를 최소화하는 방법.

- 예를 들어 0-1 loss의 경우 negative log-likelihood를 사용하여 최적화(다른 대리 함수도 많음).
  - 0-1 loss를 정확하게 최소화하는 것은 다루기 어렵기 때문.
  - 또한, 어떤 경우에는 training set의 0-1 loss가 0에 도달했음에도 불구하고 log-likelihood를 학습 시킴으로써 test set의 loss를 더 감소시킬 수 있다.

머신러닝에서의 최적화와 일반적인 최적화 사이의 또 하나의 차이점은 local minimum에서 멈추지 않는다는 것이다. 그 이유는 수렴의 표준이 early stopping을 따르기 때문이고, 따라서 gradient가 충분히 큼에도 불구하고 학습이 종료될 수도 있다.

### 8.1.3 Batch and Minibatch Algorithm

머신러닝에서의 최적화와 일반적인 최적화이 다른 이유중 하나는 목적함수를 training examples에 대한 합으로 분해한다는 것이다.

- 아래와 같은 MLE 문제를 생각해보면 최적화 해야하는 목적함수와 그에 대한 gradient는 아래와 같다.

  $$\boldsymbol \theta_{ML} = argmax_{\boldsymbol \theta}\sum^m_{i=1}\log p_{model}(\boldsymbol x^{(i)},y^{(i)},\boldsymbol \theta)$$

  $$J(\boldsymbol \theta) = \mathbb E_{\boldsymbol x,y \sim \hat p_{data}}\log p_{model}(\boldsymbol x,y;\boldsymbol\theta)\ ,\ \ \nabla_{\boldsymbol \theta}J(\boldsymbol \theta) = \mathbb E_{\boldsymbol x,y \sim \hat p_{data}}\nabla_{\boldsymbol \theta}\log p_{model}(\boldsymbol x,y;\boldsymbol\theta)$$

  - 모든 dataset을 한번에 사용(batch)하여 gradient를 계산하는 것은 cost가 너무 크다.
  - $n$개의 데이터에 대한 평균의 standard error는 $\sigma/\sqrt{n}$ 으로 주어지기 때문에, 데이터양을 늘리는 것에 비해 standard error의 감소량이 작다.
  - 따라서 standard error가 조금 크더라도 랜덤하게 샘플링된 작은 dataset를 사용(minibatch)하여 학습하는 것이 수렴이 빠름.(Minibatch의 size를 1로 두는 것을 stochastic  또는 online method라고 한다)

- Minibatch를 사용하는 또 다른 이유는 identical examples에 대한 gradient 계산량을 줄일 수 있다는 점이다. 모든 데이터 셋을 한번에 사용하면 똑같은 gradient 계산을 여러번 해야하지만, minibatch를 사용하면 수렴이 빠르기 때문에 중복된 계산을 줄일 수 있다.

- Minibatch size를 결정하는데 고려해야할 것들.

  - 크면 클수록 정확하지만 효율이 좋지 않음.
  - 멀티코어의 경우 너무 작은 batch size는 활용률이 낮다.
  - Batch size를 크게 해야할 경우 큰 메모리가 필요하다.
  - GPU를 사용할 경우 batch size는 2의 제곱수로 결정하는게 좋다.
  - 작은 batch size는 정칙화 효과를 가져올 수 있다.(Wilson and Martinez, 2003) 아마 작은 batch size로부터 발생하는 noise때문일 것으로 예상되는데, 그렇기 때문에 작은 batch size일수록 학습률을 낮춰야 한다.

- Gradient만을 사용하여 학습시키는 알고리즘의 경우 batch size는 100미만으로 작게해도 되지만, Hessian matrix가 사용되는 알고리즘의 경우는 gradient의 작은 변화로도 결과를 크게 바꿀 수 있기 때문에 batch size를 크게 키울 필요가 있다.

- Minibatch를 랜덤하게 샘플링하는게 매우 중요하다. Gradient를 unbiased estimation을 하기 위해선 데이터들이 독립이어야 하지만 많은 데이터셋이 그렇지 않기때문.

  - 매우 큰 데이터 셋의 경우, minibatch를 뽑을 때마다 데이터셋을 셔플할 필요는 없고 처음 한번만 셔플해도 충분하다고 함.

- 첫번째 에폭에서는 unbiased하게 모델이 학습되지만 두번째 에폭부터는 데이터들이 재사용되기 때문에 bias가 생길 수 밖에 없다. 그러나 전체 데이터셋을 여러번 학습함으로써 training error가 줄어드는 이득을 볼수 있기 때문에 에폭수를 늘리는 것이 좋다.(물론 early stopping이 허용하는 한에서)

## 8.2 Challenges in Neural Network Optimization

### 8.2.1 Ill-conditioning

오목 함수를 최적화 하는 과정에서 많은 문제가 발생하는데, 대표적인 예로 Hessian matrix로 인한 문제가 있다.

어떤 함수를 2차 테일러 시리즈로 근사하면 아래와 같이 쓸수 있는데,

$$f(\boldsymbol x)=f(\boldsymbol x^{(0)}) +(\boldsymbol x-\boldsymbol x^{(0)})^{\top}\boldsymbol g +\frac{1}{2}(\boldsymbol x-\boldsymbol x^{(0)})^{\top}\boldsymbol H(\boldsymbol x-\boldsymbol x^{(0)})$$

이를 SGD를 이용하여 최적화 시킨다고 하면 다음 스텝에선 $\boldsymbol x = \boldsymbol x^{(0)}-\epsilon \boldsymbol g$이므로,

$$f(\boldsymbol x^{(0)}-\epsilon \boldsymbol g)\approx f(\boldsymbol x^{(0)})-\epsilon \boldsymbol g^{\top} \boldsymbol g +\frac{1}{2}\epsilon^2 \boldsymbol g^{\top}\boldsymbol H \boldsymbol g $$

문제는 $\frac{1}{2}\epsilon^2 \boldsymbol g^{\top}\boldsymbol H \boldsymbol g$ 가 $\epsilon \boldsymbol g^{\top}\boldsymbol g$ 보다 커지는 상황이다. 따라서 이를 방지하려면 학습률 $\epsilon$이 작아져야 하고, 따라서 학습이 매우 느려진다.

위와같이 Hessian matrix로 인해 발생하는 문제를 해결하는데 좋은 방법으로 Newton's method가 있지만 이 방법 또한 뉴럴넷에 적용하기 위해선 수정이 필요하다고 한다.

### 8.2.2 Local Minima

뉴럴넷은 non-convex 비용함수를 가지기 때문에 많은 local minima를 가질 수 있는데, 사실 모든 deep model들은 무수히 많은 local minima를 가진다고 한다.

- deep model같은 경우 서로 다른 weights를 가진 모델이더라도 동일한 output을 낼 수 있기때문에(weight space symmetry) 두 모델을 식별할 수 없고 따라서 매우 많은 local minima를 가지게 된다.

local minima에서의 비용함수 값이 global minima에서의 값과 차이가 많이 난다면 큰 문제지만, 충분히 큰 모델에서는 대부분의 local minima가 낮은 비용함수 값을 가지기 때문에 별로 문제가 되지 않는다고 한다. 

그렇지만 high-dimension에선 local minima에 빠졌다는 것을 알기가 매우 어렵기 때문에 여전히 많은 사람들이 고통받고 있다.

### 8.2.3 Plateaus, Saddle Points and Other Flat Regions

- 램덤 함수(실험에 의해 구해진 값이 argument가 되는 함수)는 아래와 같은 특징을 가진다.
  - low-dimension에서는 saddle point 보다 local minima가 더 자주 발생하지만, high-dimension에서는 saddle point가 더 많이 생긴다.
    - Local minima에서는 Hessian matrix의 고유값들의 부호가 모두 양수이지만 saddle point에서는 음수 양수가 섞여있기 때문에, dimension이 높아질수록 saddle point가 발생할 확률이 더 높다.
  - local minima에선 낮은 비용함수 값을, saddle point에선 상대적으로 높은 비용함수 값을 가질 가능성이 높다.
- 실제로 뉴럴넷에서도 위와같은 특징이 발견된다고한다. (BaldiandHornik(1989), Saxe et al. (2013), Dauphin et al. (2014), Choromanska et al. (2014))
- Saddle point가 증가하면 first-order optimization이 위험할 수 있지만,  경험으로 미루어 보면 1차 최적화를 사용해도 saddle point를 잘 지나간다고 한다.
- 기울기가 0이 되는 지점은 local maxima도 있고 wide flat region도 있을수 있는데, local maxima는 극히 드물게 발생하기 때문에(local minima와 같은 이유) 문제가 되지 않지만 wide flat region같은 경우 오목 함수가 아닌 이상 문제가 어려워진다.

### 8.2.4 Cliffs and Exploding Gradients

![_config.yml](C:/Users/astro/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch8/Fig8_3.png)

뉴럴넷에서는 종종 위와같은 기울기 절벽(?) 이 발생하는데, 경사하강법을 그대로 쓰게되면 이상한 위치로 튈 수 있기 때문에 weight clipping을 해줘야 한다. 이러한 절벽은 RNN에서 자주 발생한다고 한다.

### 8.2.5 Long-Term Dependencies

Computational graph가 너무 깊어져도 문제가 발생한다. 특히 RNN의 경우, 동일한 연산이 반복되기 때문에 더 취약하다. Matrix $\boldsymbol W$를 곱하는 연산이 연속적으로 $t$번 행해지는 computational graph가 있다고 하면 결과적으로 $\boldsymbol W^t$가 곱해진다. $\boldsymbol W$를 eigen-decomposition하게 되면 $\boldsymbol W^t = \boldsymbol V \text{diag}(\boldsymbol\lambda)^t\boldsymbol V^{-1}$ 이 되는데, 이때 $\boldsymbol\lambda$가 1보다 큰지 작은지에 따라 기울기가 explode하거나 vanish한다.

### 8.2.6 Inexact Gradients

대부분의 최적화 알고리즘은 정확한 기울기나 정확한 Hessian matrix를 알고있다는 가정에서 시작하는데, 현실에서 최적화를 진행할 때는 여러가지 이유로 minibatch를 사용한다. 즉, 기울기가 biased 되어있다. 또한 기울기를 구할 수 없는 경우도 발생한다. 이런 경우 대체 비용함수를 사용해서 해결 가능하다.

### 8.2.7 Poor Correspondence between Local and Global Structure

![_config.yml](C:/Users/astro/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch8/Fig8_4.png)

지금까지는 한 점에서 발생하는 문제들을 봤는데(local minima, saddle point ...), 이 모든 문제점들을 해결해도  발생하는 문제가 있다.

- 위의 그림에선 local minima나 saddle point가 없음에도 불구하고 최적화가 안되는 것을 볼 수 있는데, 이러한 문제가 발생하는 이유는 대부분의 최적화 알고리즘이 small, local move를 사용하기 때문이다.
- 현실에서는 inexact gradient를 사용하기 때문에 올바른 방향으로 학습이 되고있는지도 장담할 수 없다.
- ㄴㄴ또한 굉장히 비효율적인 길을 따라 최적화가 진행될 수도 있다.

아직 비용함수의 global structure를 사용하거나 non-local move를 이용하는 알고리즘에 대한 연구가 덜 되어서 대부분 initial condition을 조정해가며 해결한다고 한다.

### 8.2.8 Theoretical Limits of Optimization

- 몇몇 연구에서는 이론적 결과는 오직 이산값을 주는 신경망에만 적용 가능하다고 하지만, 대부분의 신경망은 로컬 검색을 통해 최적화 할 수 있도록 smooth하게 증가하는 값을 출력한다.
- 어떤 연구결과에서는 네트워크 사이즈가 정해져있을 때 솔루션을 찾는 것은 매우 어려운 문제라고 한다.
  - 현실에서는 그냥 네트워크 사이즈를 키워서 해결할 수 있고, 정확한 minimum을 찾지 않아도 충분히 좋은 성능을 낼 수 있기 때문에 큰 문제가 아니다.

최적화 알고리즘에 대한 이론적 분석은 매우 어려운 작업이지만 최적화 알고리즘의 성능에 대한 현실적인 bound에 대한 연구는 중요한 과제이다.



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

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-rmsprop-nesterov.png)

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

![_config.yml]({{ site.baseurl }}/assets/ch8/converge-newton.png)

- 이 알고리즘은 최대 경사법 (책 4.3절) 의 문제를 보완한 알고리즘인데, 최대 경사법의 문제는 다음과 같다.
    - $t_0$에서 line search 방향이 $d_{t_0}$라고 하면, $t_0$가 끝난 후 $d_{t_0}$ 방향으로의 기울기는 0이다.
    - $t_1$에서 $d_{t_0}$방향의 기울기가 0이기 때문에 $d_{t_1}$과 $d_{t_0}$는 직교한다. 이 때문에 위 그림처럼 계단형 궤적이 생긴다.
    - 문제는 $t_0$에서 개선한 것을 $t_1$에서 일정부분 다시 되돌린다.
- 이러한 문제를 해결하기 위해 켤레 기울기법에서는 이전 방향과 켤레인 방향을 선택한다 (아래 식의 $\beta_t$를 잘 선택하여 켤레인 방향을 만들어야 함). 이렇게하면 이전 방향 목적함수의 최솟값을 유지하며 다음 업데이트를 할 수 있다.

$$d_t = \nabla_\theta J(\theta) + \beta_t d_{t-1}$$

- 문제는 $d_t^T H d_{t-1}$의 고윳값을 찾아서 $\beta_t$를 찾아야 한다는 점이다. 이렇게 할 경우 계산 비용 면에서 뉴턴법과 별 차이가 없다. 다행히도 $H$를 계산하지 않고도 $\beta_t$를 계산할 수 있는 방법들이 있다.
    - Flather-Reeves: $\beta_t = \frac{\nabla_\theta J(\theta_t)^T\nabla_\theta J(\theta_t)}{\nabla_\theta J(\theta_{t-1})^T\nabla_\theta J(\theta_{t-1})}$
    - Polak-Ribiere: $\beta_t = \frac{(\nabla_\theta J(\theta_t)-\nabla_\theta J(\theta_{t-1}))^T\nabla_\theta J(\theta_t)}{\nabla_\theta J(\theta_{t-1})^T\nabla_\theta J(\theta_{t-1})}$

![_config.yml]({{ site.baseurl }}/assets/ch8/algo-conjugategrad.png)

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

## 8.7 Optimization Strategies and Meta-Algorithms

### 8.7.1 batch normalization

adaptive reparametrization(재매개변수화) 방법중 하나. 아주 깊은 모형의 훈련에서 발생하는 문제들을 막아준다. 

하나의 GD 과정의 weight update는 다른 층이 변하지 않는다는 가정 하에서 이루어진다. 그런데 일반적인 학습 과정에서는 모든 층을 동시에 갱신한다. 이에 따라 예기치 않은 결과가 발생할 수 있다. 

예를 들어 어떤 신경망을 다음과 같이 표현해보자. 

$$\hat{y}= xw_1w_2w_3...w_l$$

GD로 얻은 cost function의 기울기가 1일 때 backprop을 통해 이전 층의 기울기 $g = \nabla_w \hat{y}$를 계산한다. 이때 $w$ ← $w - \epsilon g$라는 업데이트를 각 층에 대응해 업데이트된 y를 계산하면 아래와 같다. 

$$x(w_1 - \epsilon g_1)(w_2 - \epsilon g_2)(w_3 - \epsilon g_3)...(w_l - \epsilon g_l)$$

이런 갱신에서 나오는 2차 항들중엔 $\epsilon^2 g_1 g_2 \prod^l_{i=3}w_i$와 같은 항들이 나오는데, $\prod$값이 작으면 무시가능하고 그게 아니면 지수적으로 이 항의 크기=오차가 커진다. 

![_config.yml]({{ site.baseurl }}/assets/ch8/Untitled.png)

batch normalization은 거의 모든 NN을 reparametrize할 수 있는 방법을 제공한다. H를 minibatch를 담은 design matrix라고 할 때 H를 normalize하는 식은 아래와 같다. 

$$H' = {H-\mu \over \sigma}$$

$$\mu = {1 \over m} \sum_i H_{i, :},\ \ \  \sigma = \sqrt{\delta + {1 \over m}\sum_i(H-\mu)^2_i}$$

이렇게 minibatch를 한꺼번에 학습시키고 평균을 내 GD를 수행하면 위에서 언급한 오차(2차항들)이 제거되며 낮은 층들의 출력은 unit gaussian으로 reparametrize된다. 

이렇게 normalize가 수행되면 Neural network의 expressive power(표현력)이 줄어들 수 있다. 이때 minibatch $H$를 $H'$로 normalize하는 대신 $\gamma H' + \beta$로 대체하는 것으로 이를 막을 수 있다. $\gamma, \beta$는 학습되는 parameter들로서 새 parameter들이 적절한 평균과 표준편차를 갖도록 만든다. 

batch normalization 오리지널 논문에서는 각 layer에서 activation function을 통과하기 이전에 batch normalization을 수행했으나 activation funtion 이후에 batch norm을 수행하는 것이 더 좋다는 결과도 있다. activation funtion 이후에 수행하면 좀 더 명확하게 통계치를 컨트롤할 수 있기 때문으로 보인다. 

### 8.7.2 Coordinate Descent(좌표 하강법)

f(x)를 한 변수 $x_i$에 대해 최소화하고, 그다음 i+1번째 변수에 대해 최소화하하는 식으로 모든 변수에 대해 차례로 최소화를 한다면 반드시 local minima에 도달할 수 있게 된다. 이런 방법을 Coordinate Descent이라고 한다. 하나씩이 아니라 정해진 block별로 최소화를 진행하면 block coordinate descent라고 한다. 

coordinate descent는 여러 변수들을 분별된 그룹으로 깔끔하게 분리가 가능한 문제가 있을 때 유용하다. 반면 한 변수가 다른 변수의 최적값에 강하게 영향을 미치는 경우엔 좋은 전략이 아닐 것이다. 

### 8.7.3 Polyak Averaging

최적화 알고리즘이 매개변수 공간을 거쳐 간 자취에 있는 여러 점의 평균을 구하는 방법. GD를 t번 반복해 각각 얻은 parameter set을 $\theta^{(1)}, \theta^{(2)},...\theta^{(t)}$라고 한다면 Polyak Averaging을 통해 얻은 결과는 아래와 같다. 

$$\hat{\theta}^{(t)} = {1 \over t}\sum \theta^{(i)}$$

몇몇 문제에서 이 방식으로 최적화를 수행하면 convergence가 보장된다. 최적화 알고리즘이 계곡에 다다르지 못하고 그 주변을 맴돌 때 이 궤적을 평균화해서 계곡 안까지 보내는 것이다. nonconvex문제에서는 cost function이 지나는 궤적이 매우 복잡해 현재 이터레이션에서 멀리 떨어진 과거의 파라미터들을 사용하는 것이 부적절할 수 있으므로 아래와 같이 지수적으로 감소하는 weighted average를 사용한다. 

![_config.yml]({{ site.baseurl }}/assets/ch8/Untitled 1.png)

### 8.7.4 Supervised Pretraining

모형이 복잡하거나 최적화가 힘들 때 그 모형을 직접 훈련하지 않고 더 간단한 문제를 푼 다음 그 모형을 더 복잡하게 만드는 것이 가능하다. 더 간단한 task에 대해 간단한 model을 훈련하는 방식을 pretraining이라고 한다. 

Greedy supervised pretraining은 하나의 supervised 문제를 더 간단한 여러 supervised 문제로 바꾸는 방법론을 지칭한다. 

![_config.yml]({{ site.baseurl }}/assets/ch8/Untitled 2.png)

위 그래프가 Greedy supervised pretraining의 예시이다. (a), (b)는 얕은 model을 훈련하는 것을 뜻하며, (c), (d)에서는 (a)의 출력부를 제거하고 한 층을 더 쌓아 훈련을 진행한다. 

왜 잘 될까? 중간의 은닉층들에 더 좋은 guide를 제공하기 때문이라고 원 논문[Bengio 2017]은 설명한다. pretraining은 optimization & generalization에 도움이 된다. [Yosinski et al. (2014)]에선 transfer learning의 맥락에서 이를 하용하여, 분리된 데이터셋으로 얕은 NN과 확장된 NN을 각각 학습시킨다. FitNets[Romero 2015]에서는 얕고 넓은 NN을 학습한 후 이를 teacher로 삼아 student 역할을 하는 가늘고 깊은 NN을 학습시킨다. 여기서 student NN은 teacher NN의 중간 층 예측도 추가적인 과제로 수행해야한다. 이로 인해 가늘고 깊은 신경망이 얕고 넓은 신경망의 특성도 배울 수 있다 .

### 8.7.5 Designing Models to Aid Optimization

강력한 최적화 알고리즘을 사용하는 것보다 최적화하기 쉬운 model family를 선택하는 것이 더 중요하다. NN 학계에서 일어난 진보는 대부분 최적화의 영역보다 model family의 발전이었다. (장 마지막에 김빠지게 이런 말을 하냐)

LSTM, maxout unit 등 최근에 나온 혁신적인 모형들은 과거에 비해 선형함수에 가까워졌고, 최적화가 쉽다는 성질을 지닌다.

최적화를 위한 다른 설계전략들도 많다.  skip connection이라는 모델을 사용하면 전체 경로가 줄어 gradient vanishing 문제가 해결된다. output의 복사본을 NN의 중간 hidden layer에 부착하는 방법도 있다.(googlenet) 

![_config.yml]({{ site.baseurl }}/assets/ch8/Untitled 3.png)

### 8.7.6 Continuation Methods and Curriculum Learning

8.2.7에서 소개했듯 optimization의 어려움은 cost function의 global structure에 기인한다. 이를 해결하기 위해 Continuation method를 사용한다. 

Continuation method(연속법)는 optimization 과정이 적절한 영역에서 수행되도록 initial point를 잘 세팅하는 방법을 말한다. Continuation method의 핵심은 parameter에 대한 objective function을 잘 세팅하는 것이다. 최종 과제에 대한 cost func를 $J(\theta)$라고 하면, Continuation method에서는 여러개의 cost function($J^{(0)}, J^{(1)}, ...,J^{(n)}$)을 만든다. 각 cost function은 index가 증가할수록 optimize가 어렵게 설계되어있다. 즉 $J^{(0)}$이 가장 minimize가 쉽고, $J^{(n)}$이 가장 어렵다. 

최소화가 쉽다는 것은 $\theta$ 공간에서 더 바람직하게 행동한다는 뜻이며, 이를 최소화하는 과정에서 본래 cost function $J(\theta)$가 더 잘 최적화되는 시작점을 설정할 수 있다. 

local minima의 영향에서 벗어나 global minima에 도달하기 위해 원래의 cost function을 흐리게(blurring)만드는 방법을 사용한다. 

![_config.yml]({{ site.baseurl }}/assets/ch8/Untitled 4.png)

이 방식은 non-convex를 흐리게 하면 convex func가 된다는 가정에서 출발한다. 

Curriculum learning 혹은 shaping은 먼저 간단한 개념을 배우고 그 개념들에 의존하는 더 복잡한 개념들의 학습으로 나아가는 방법을 뜻하며, 이는 방금 소개한 Continuation Methods를 통해 설명할 수 있다. 더 쉬운 J들은 더 simpler examples의 영향을 증가시켜(덜-숨겨진 feature?) 이후 cost function의 최소화를 쉽게 만든다. 더 쉬운 J를 teacher라고 한다면, teacher은 더 쉽고 기본적인 예들을 보여주고, 학습자가 덜 명확한 사례들에 대해 적절한 decision boundary를 세팅하는 것을 돕는다. 

쉬운 example들과 어려운 example들을 섞되 평균적으로 더 어려운 example이 등장하도록 하는 stochastic curriculum method는 일반적인 curriculum learning보다 더 나은 성과를 냈다.
