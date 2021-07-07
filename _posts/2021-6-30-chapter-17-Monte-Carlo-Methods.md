몬테카를로 방법에 대해 소개하는 장

- 무작위화 알고리즘(randomized algorithm)은 크게 라스베가스, 몬테카를로 알고리즘 두 가지로 나뉨
- 라스베가스 알고리즘은 항상 정확히 맞는 답을 내놓거나, 실패했다고 보고함
  
  - 이로 인해 랜덤한 양의 자원(메모리, 시간 등)을 사용함
  
- 반면, 몬테카를로 알고리즘은 랜덤한 양의 에러가 더해진 대략적인 답을 내놓음
  
  - 에러의 양은 일반적으로 많은 자원을 사용할수록 줄어듬
- 기계학습에서 마주치는 정확한 답을 맞추기에 너무 어려운 문제의 경우엔, 정확한 결정적인(deterministic) 라스베가스 알고리즘 보다는, 대략적인(approximate) 몬테카를로 알고리즘을 쓰는 것이 좋음
  
  - 두 방법 모두 기계학습에서 널리 사용되는데, 이번 단원에서는 몬테카를로 방법에 초점을 둠


## 17.1 Sampling and Monte Carlo Methods

- 기계학습의 많은 기법들은 어떠한 확률 분포에서 샘플을 만들고, 이를 이용해 원하는 지표의 몬테카를로 추정(estimate)를 얻음


### 17.1.1 Why Sampling?

- 확률 분포에서 샘플을 뽑는 이유로, 작은 연산 양으로도 많은 덧셈이나 적분의 값을 유동적으로 가늠할 수 있게함

  - 그렇지 않다면 알고리즘에 포함되는 수많은 복잡한 덧셈과 적분 값을 일일이 계산해야함
    
  - 예) 일부 minibatch의 학습 비용을 샘플링해서 대략적인 평균 수치를 가늠함

  - 따라서 학습 분포에서 샘플링 할 수 있도록 모델을 학습시키는 것은 많은 기계학습 문제의 목표임


### 17.1.2 Basics of Monte Carlo Sampling

- 덧셈이나 적분이 항의 개수가 너무 많은 등의 이유로 정확히 계산될 수 없을 때, 몬테카를로 샘플링으로 얻은 평균으로 근사값을 구할 수 있음

  - 적분 $s$에 대해, 확률분포 $p$로부터 $n$개의 샘플 $x^{1}$, ..., $x^{n}$을 샘플링해 평균값을 구함

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_1_2.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_3.PNG)

  - 특징 1) Estimator $\hat{s}$는 unbiased 되어있음

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_4.PNG)

  - 특징 2) 큰 수의 법칙(law of large numbers)에 의해 만약 $x^i$가 i.i.d. 라면 평균 값은 기대값에 수렴하므로, $\hat{s}_{n}$의 분산은 $n$이 커질수록 감소함 (단, $f(x^i)<\infty$ 일 때) 

    - 큰 수의 법칙: 큰 모집단에서 무작위로 뽑은 표본의 평균이 전체 모집단의 평균과 가까울 가능성이 높다는 통계와 확률 분야의 기본 개념

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_5.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_6_7.PNG)

- 한 편, 확률분포 $p(x)$에서 샘플링하는 것이 어려울수도 있음

  - 대신 사용할 수 있는 방법으로 importance 샘플링이 있음 (17.2에서 소개)

  - 보다 일반적인 방법으로는 몬테카를로 마르코브 체인이 있음 (17.3에서 소개)


## 17.2 Importance Sampling

- $p(x)$, $f(x)$를 각각 얻어낼 수 없을 때, 아래와 같은 식으로 식을 변형해 $q$, $pf/q$를 샘플링 할 수도 있음

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_8.PNG)

- Importance sampling estimator의 기대값은 유지되어 $q$에 영향을 받지 않음

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_9.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_10.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_11.PNG)

- 하지만 분산은 $q$를 어떻게 선택하는지에 따라 예민하게 바뀜

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_12.PNG)

  - 따라서 분산을 최소화하도록 최적화된 $q$는 아래와 같음 ($Z$: normalization constant)

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_13.PNG)

- 만약 $f(x)$의 부호가 바뀌지 않는다면, $Var[\hat{s}_{q*}]=0$ 인 경우 단 하나의 샘플이면 충분하다는 뜻이됨

  - 단, 이는 $q*$를 선택함으로써 원래의 문제가 완전히 풀려버렸다는 뜻으로, 일반적으로 발생하는 경우는 아님

- $p$, $q$가 정규화되어 있지 않다면, biased importance sampling이 됨 ($\tilde{p}, \tilde{q}$: Unnormalized form, $\mathbb{E}(\hat{s}_{BIS}) \neq s$)

![_config.yml]({{ site.baseurl }}/assets/ch17/Eq17_14_16.PNG)

- $q$를 잘 선택하면 몬테카를로 샘플링의 효율을 크게 향상 시킬 수 있지만, 잘못 선택하면 오히려 안 좋아짐

  - $p \mid f \mid /q$ 항을 고려하면, $q$가 $p$, $f$보다 훨씬 작을 때 분산이 커지게 됨

  - 한 편, $q(x)$가 너무 커지면, $pf/q$가 0이나 너무 작은 수가 되어 쓸모없는 샘플들을 뽑게됨

- 이러한 장애물이 있음에도 기계학습의 전반적인 분야에서 널리 쓰이는 기술임 (많은 수의 단어에 대한 언어 모델 등)

  - 18, 20단원에서 자세히 소개 될 예정임



## 17.3 Markov Chain Monte Carlo Methods

### 17.3.1 What is Markov Chain?

Markov Chain은 특정 step에 대응하는 상태의 확률이 이전 step의 상태에만 의존하는 discrete probabilistic process를 뜻한다. 

![_config.yml]({{ site.baseurl }}/assets/ch17/Untitled.png)

날씨를 맑음 / 흐림으로 구분하고, 내일의 날씨가 오늘의 날씨에만 직접적으로 연관이 있다는 가정을 했을 때,  내일의 날씨를 위와 같이 오늘의 날씨에 기초한 전이 확률로 나타낼 수 있다. 내일의 날씨가 오늘 날씨에만 의존하므로 Makov process이며, 좀 더 엄밀한 표현으로는 memory k=1인 마르코프 프로세스이다. 

### 17.3.2 MCMC

Monte Carlo technique를 사용하고 싶지만, $p_{model}(x)$ 혹은 적절한 sampling distribution $q(x)$에서 샘플링을 하는 것이 어려운 경우가 많다. 특히  $p_{model}(x)$이 undirected model인 경우 이런 경우가 많이 발생한다. 이 때, markov chain이라는 방법을 이용해  $p_{model}(x)$를 sampling할 수 있다. 이런 방법을 Markov chain Monte Carlo method(MCMC)라고 한다. 이 방법론은 0의 확률이 있는 상태가 없는 분포에서만 사용이 가능하다. 따라서 Energy based model( $p(x) \sim exp(-E(x))$에서 표본을 추출하는 방법을 사용하면 다루기 편하다. 

Energy based model에서 sampling을 진행하기 어려운 이유는 다음과 같다. 어떤 분포 *p*가 a, b를 변수로 가질 때, a를 샘플링하려면 $p(a\mid b)$에서 a를 뽑아야 한다. 반대로 b를 sampling하려면 $p(b\mid a)$에서 b를 뽑아야 한다. 이런 상호참조의 문제가 생긴다. directed model의 경우엔 acyclic한 그래프를 사용하므로 이런 문제가 없다. 

Energy based model에서 Markov chain(MC)을 사용하면 상호참조 문제를 해결할 수 있다. Markov chain 방법에서는 상태 x를 임의의 값으로 초기화한 후 이를 갱신하는 방법을 사용한다. 갱신이 반복되면 x는 p(x)의 fair한 sample에 가까워진다. 

MC는 random state x와 transition distribution(전이 분포) $T(x' \mid x)$으로 구성된다. transition distribution은 x가 x'이 될 확률을 알려준다. MC의 각 단계의 상태는 $q^{(t)}(x)$에서 추출되며, t는 지금까지 거쳐 온 시간 단계의 수이다. 시작 저점에서 $q^{(0)}$을 이용해 각 MC의 x를 임의의 값으로 초기화되며, 이후 이 q는 계속 업데이트가 된다. 목표는 $q^{(t)}(x)$가 $p(x)$에 수렴하게 하는 것이다. 

확률 분포 q를 vector v를 이용해 아래와 같이 서술할 수 있다. 

$$q(X=i)=v_i.$$

어떤 timestep t에서 q가 주어졌을 때, t+1에서는 아래와 같이 q의 갱신이 이루어진다. 

$$q^{(t+1)}(x') = \sum_x q^{(t)}(x)T(x'\mid x).$$

parameter들이 정수일 때 transition distribution을 행렬 A로 나타낼 수 있다. 그렇다면 A는 아래와 같이 정의된다. 

$$A_{ij} = T(x'=i \mid x=j).$$

v와 A를 이용해 q의 갱신식을 다시 쓰면 아래와 같다. 

$$v^{(t)} = Av^{(t-1)}.$$

이 과정을 여러번 반복하면 아래와 같은 식이 도출되고, 

$$v^{(t)} = A^t v^{(0)}$$

만약 A라는 transition matrix, v, v'가 새로운 time step에서 같은 값을 가진다면, 다시말해 아래와 같은 식이 도출된다면, 

$$v' = Av = v$$

수렴한다고 할 수 있다. 실제 MCMC를 사용하는 상황에서는, 특정 threshold 이하로 오차가 나오면 update를 종료한다. MC가 stationary state에 도달할 때 까지의 단계 수를 미리 알 수 없는 점이 하나의 단점이다. 

## 17.4 Gibbs sampling

지금까지 x←x' ~ $T(x' \mid x)$를 계속 갱신해 q(x)로부터 sampling을 하는 방법을 논의했다. 이제 적절한 q(x)를 택하는 문제에 대해 알아본다. 이 책은 두 접근방식을 고려한다. 

1. 주어진 학습된 분포 $p_{model}$으로부터 T를 유도하기 → 이 장에서 설명
2. T를 직접 매개변수화해 해당 분포가 $p_{model}$을 내재적으로 정의하도록 매개변수들을 학습시키기

Gibbs Sampling을 이용하면 $p_{model}$에서 sampling을 할 때 markov chain을 효과적으로 구축할 수 있다. Gibbs Sampling은 두 개 이상의 확률변수의 결합확률분포로부터 일련의 표본을 생성하는 확률적 알고리즘이다.

Gibbs Sampling의 절차는 아래와 같다. 

1. 임의의 표본 $𝑋^0=(𝑥^1,𝑥^2,𝑥^3)$을 선택한다.
2. 모든 변수에 대해 변수 하나만 변경해 새로운 표본 $X^1$을 뽑는다. 

두번째 과정을 좀 더 상세히 살펴보면 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch17/Untitled1.png)



## 17.5 The Challenge of Mixing between Separated Modes

MCMC를 사용할 때 가장 곤란한 점은 두 'mode'가 잘 섞이지 않는다는 것이다(특히 차원이 높은 공간에서 더 취약하다). 

- 마르코브 체인을 통해 샘플링 된 변수 $\boldsymbol x^{(t-1)}$은 에너지 $E(\boldsymbol x^{(t-1)})$에 따라 $\boldsymbol x^{(t)}$로 이동하는데, 일반적으로 $E(\boldsymbol x^{(t)})$는 $E(\boldsymbol x^{(t-1)})$보다 작거나 같기 때문에 $\boldsymbol x^{(t)}$는 점점 에너지가 낮은 영역으로 이동하게 된다.
- 이렇게 에너지가 낮은 영역을 'mode'라 부르고 상대적으로 에너지가 낮기 때문에 에너지 장벽에 막혀서 다른 mode로 넘어가기 힘들다.
  ![_config.yml]({{ site.baseurl }}/assets/ch17/Fig17_1.png)
- 위의 그림처럼 하나의 mode만 있는 경우에는 크게 문제가 되지 않는다.(다만, 변수들의 correlation이 높을수록 초기 위치에서 벗어나기 어렵다.) 하지만, 세번째 그림과 같이 mode가 여러개인 경우에는 처음 정해진 mode에서 벗어나기 어려운 것을 볼 수 있다.

이런 현상은 잠재 변수 $\boldsymbol h$를 가지는 모델에서도 자주 일어난다.

확률분포 $p(\boldsymbol x, \boldsymbol h)$ 에서 $\boldsymbol x$라는 변수를 샘플링 하기 위해 $p(\boldsymbol x\vert \boldsymbol h)$와 $p(\boldsymbol h\vert \boldsymbol x)$에서 번갈아 가며 샘플링하는 경우가 많다. 여러 mode가 잘 섞이기 위해서는 $p(\boldsymbol h\vert \boldsymbol x)$의 엔트로피가 높아야 하고, $\boldsymbol h$가 좋은 representation을 가지려면 $\boldsymbol x$와 $\boldsymbol h$ 사이의 상호의존정보가 높아야 할 것이다. 하지만 이 둘은 동시에 일어날 수 없다.

![_config.yml]({{ site.baseurl }}/assets/ch17/Fig17_2.png)

왼쪽 그림처럼 볼츠만 머신을 사용한 경우에는 슬로우 믹싱이 일어난 것을 볼 수 있다.

### 17.5.1 Tempering to Mix between Modes

슬로우 믹싱은 확률분포가 뾰족할 때 잘 일어난다. 이때 '온도'라는 개념을 사용하면 좀 더 정확한(다양한) 샘플링이 가능하다. 기존의 에너지 기반 모델의 확률 분포는 아래와 같이 정의 되었는데

$$p(\boldsymbol x) \sim \exp(-E(\boldsymbol x))$$

여기에 추가적인 파라미터 $\beta$(온도에 반비례)를 도입하여 아래와 같이 정의 할 수 있다.

$$p(\boldsymbol x) \sim \exp(-\beta E(\boldsymbol x))$$

즉, 온도가 낮으면 확률 분포가 더 뾰족하고, 온도가 높을 수록 평평 하다. 

온도를 이용하여 mode사이의 믹싱이 일어 날 수 있도록 해주는 여러가지 방법이 존재한다.

- Tempered transition(Neal, 1994) : 높은 온도에서 시작하여 여러 mode 사이의 믹싱이 일어나게 한 후에 온도를 1로 조정하여 샘플링하는 방법.

- Parallel tempering(Iba, 2001) : 서로 다른 state들을 서로 다른 온도에서 병렬로 마르코브 체인을 수행하고, 각 iteration마다 메트로 폴리스 알고리즘에 따라 서로 다른 온도의 샘플을 서로 바꾼다.

  ![_config.yml]({{ site.baseurl }}/assets/ch17/Parallel_tempering.png)

  ([https://www.researchgate.net/figure/Schematic-of-the-parallel-tempering-method-for-fitting-the-model-In-parallel-tempering_fig6_276922924](https://www.researchgate.net/figure/Schematic-of-the-parallel-tempering-method-for-fitting-the-model-In-parallel-tempering_fig6_276922924))

### 17.5.2 Depth May Help Mixing

잠재 변수를 사용하는 모델을 사용할 때 $\boldsymbol x$부터 $\boldsymbol h$까지의 깊이를 깊게하면 믹싱에 도움을 준다고 한다. 두 샘플을 인코딩 했을 때 만들어진 두 잠재 변수는 서로 떨어져 있어야 구분이 가능하기 때문에 많은 샘플들을 구분하기 위해서는 $\boldsymbol h$의 분포가 잠재 변수 공간에 고르게 분포하여야 한다. 이때 인코딩의 깊이가 깊어질수록 더 고르게 분포하는 경향이 있고, 이는 잠재 변수 공간에서의 마르코브 체인의 믹싱이 잘 일어나게 도와준다.