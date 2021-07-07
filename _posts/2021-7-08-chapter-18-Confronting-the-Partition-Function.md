 이 장에서는 처리 불가능한 partition function들을 어떻게 다루는지에 대해 다룬다. 

undirected graphical model중에는 unnormalized probability distribution으로 정의되는 것들이 많다. unnormalized prob $\tilde{p}$를 아래와 같이 partition function $Z(\theta)$를 사용해 normalize시킬 수 있다. 

$$p(x;\theta) = {1 \over Z(\theta)}\tilde{p}(x;\theta)$$

이 표준화를 위해 $Z$는 모든 가능한 $\tilde{p}$를 더하는 형태로 구해진다. 연속 변수는 적분 $\int \tilde{p}(x;\theta)$이고, 이산변수는 합이 될 것이다. 그런데 일반적으로 여러 유용한 모형들에서 이런 계산은 intractable(계산 불가)한 경우가 많다. 모든 경우의 수를 다 고려할 수가 없기 때문이다.

## 18.1 The Log-Likelihood Gradient

Undirected Model에서는 MLE에 기초한 학습이 어려운데, partition function들이 parameter에 의존하기 때문이다. log likelihood의 parameter에 대한 gradient를 구해보면, partition function의 gradient에 해당하는 항이 있다. 

$$\nabla_{\theta}\log p(x;\theta) = \nabla_{\theta}\log \tilde{p}(x;\theta) - \nabla_{\theta}\log Z(\theta)$$

우변의 첫번째 항을 positive phase, 두번째 항을 negative phase라 부르는데, undirected model에서는 두번째 항을 계산하기 어렵다. 예를 들어 RBM이 있다. hidden unit들의 visible unite들은 서로 조건부 독립이기 때문에 positive phase를 계산하는 것이 어렵지 않다. negative phase를 좀 더 살펴보자

$$\nabla_{\theta}\log Z = {\nabla_{\theta}Z \over Z} = {\nabla_{\theta} \sum_x \tilde{p}(x) \over Z} = {\sum_x \nabla_{\theta} \tilde{p}(x) \over Z}$$

모든 x에 대해 p(x)>0이 보장되는 모형들에서는, 즉 확률이 0인 곳이 없는 모형에서는 $x$ 대신 $\exp ( \log \tilde{p}(x))$를 대입해 수식을 더 간단하게 만들 수 있다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled.png)

결국 이 항등식은 처리 불가한 partition function을 가진 mode의 likelihood를 근사적으로 최적화하는 Monte Carlo method의 한 형태가 된다. MLE에 대한 MCMC의 접근 방식을 모형 분포를 자료가 발생한 지점으로 밀어 올리려는 힘( positive phase)와 끌어 내리는 힘(positive phase) 사이 균형을 맞추는 과정이라고 볼 수 있다. 아래 그래프가 이에 대한 모식도이고, 각각 $\log \tilde{p}$의 최대화와 $\log Z$의 최소화에 대응된다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled%201.png)

## 18.2 Stochastic Maximum Likelihood and Contrastive Divergence

18.1에서 도출한 아래 식을 실제로 구현할 때, 새로운 gradient가 필요할 때마다 markov chain을 초기화하여 혼합하는 형태가 된다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled%202.png)

아래는 이를 구하기 위한 MCMC 알고리즘이다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled%203.png)

이러한 절차는 비용이 높기 때문에 좀 더 효율적인 근사 알고리즘이 필요하다. 위 알고리즘에서 가장 코스트가 높은 부분은 무작위로 초기화한 MC를 연소하는데 드는 비용이다. 가장 간단한 최적화 깁법은 모형 분포와 비슷한 분포를 이용해 MC를 초기화하는 것이다. 이렇게 하면 MC 초기화에 드는 loop의 사이즈가 줄어든다. 

Contrastive divergence 알고리즘은 각 단계에서 자료 분포에서 추출한 표본들로 마르코프 연쇄를 초기화한다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled%204.png)

얕은 학습에서는 Contrastive divergence 방법이 잘 동작하지만, 깊은 모델에서는 잘 동작하지 않는다. visible node들의 sample들이 주어졌을 때, hidden unit들의 sample을 얻기 어렵기 때문이다. 

CD의 문제를 해결하는 다른 전략으로, 각 gradient descent 단계에서 Markov chain의 상태를 이전 gradient descent의 것들을 이용해 초기화하는 것이다. 이를 Stochastic maximum likelihood(SML)이라고 부른다. 이를 더 발전시켜 Persistent contrastive divergence(PCD)라는 방법이 소개되었다. PCD 알고리즘은 아래와 같다. PCD의 기본 아이디어는, Stochastic Gradient descent의 갱신 단계가 작다면 이전 단계의 모형이 현재와 비슷하다는 것이다. 

![_config.yml]({{ site.baseurl }}/assets/ch18/Untitled%205.png)

학습 속도를 빠르게 하는 다른 방법으로, Monte Carlo sampling 방법을 바꾸는 것이 아니라 Cost function의 parameter을 조정하는 방식도 존재한다. Fast PCD 방식은 model의 parameter을 아래와 같이 대체한다. 

$$\theta = \theta^{fast} + \theta^{slow}$$

빠른 매개변수들은 learning rate을 높여 학습시키고, 느린 매개변수들은 learning rate을 느리게하여 학습시킨다.