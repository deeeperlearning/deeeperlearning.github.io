chapter3: 확률과 정보 이론

# chapter 3: Probability and Information Theory
확률론은 과학과 공학의 여러 분야에서 근본적인 도구로 쓰인다. 확률론을 이요하면 북확실한 명제를 서술할 수 잇으며, 불확실성이 존재하는 상황에서 뭔가를 추론할 수 있다. 

정보이론을 이용하면 주어진 확률분포에 존재하는 불확실성의 양을 추정할 수 있다. 

## 3.1 Why Probability?

- 기계학습은 불확실한 수치들을 다루며 확률적, 비결정론석 수치를 다룰 때도 있다.
- 불확실성을 다루는 세가지 원천
    - **모형화할 시스템에 내재한 확률성**. 예를 들어 양자역학. , 카드들이 무작위 순서로 완저히 섞인다고 가정하는 카드겡미
    - **불완전한 관측 가능성.** 결정론적 시스템이어도 행동의 모든 변수들을 관측할 수 없는 경우
    - **불완전한 모형화.**  관측한 정보 일부가 소실되는 경우. 예를 들어 float64 data가 int16이 된다면 정보가 많이 소실됨.

- 간단하지만 불확실한 규칙이 더 나을 때가 많음.
- 반복하기 어려운 명제들에 "반복의 비율이 확률임"을 대입하는 게 어려울 때가 있다. 의사가 환자를 진단하는 경우 등.(평행우주가 존재하지 않는 이상 반복 불가)
- 이 경우 확률은 사건의 비율이 아니라 degree of belief(믿음의 정도)임.
- 전자의 확률: frequentist probability, 후자를 bayesian probability라고 한다.
- 확률을 "논리를 불확실성까지 다룰 수 있도로 확장한 것"이라고 봐도 좋다. 확률론은 다른 명제들의 likelihood(가능도)가 주어졌을 때 어떤 명제가 참일 가능도를 결정하는 형식 규칙을 제공.

## 3.2 random variable(확률 변수)

- random variable은 여러 값을 무작위로 가지는 변수. 변수 자체는 영문 소문자, 값은 이탤릭 소문자로 표기.
- random variable은 discrete일 수도 있고, continuous일 수도 있음.

## 3.3 Probability distribution

- 하나의 random variable, 혹은 random variable의 집합이 각각의 상태를 가질 likelihood를 정의.
- 이산/연속 여부에 따라 식 전개가 상이함

### 3.3.1 Descrete variables, Probability mass function

- Descrete variable을 서술하는 방법으로 probability mass function(PMF)를 사용, 대분자 P
- P(x=$x)$와 같이 특정 변수값에 대한 확률 수치를 나타낼 수 있음.
- PMF의 조건:
    - P의 domain(정의역)은 x의 모든 가능한 상태의 집합이어야 함.
    - $\forall x \in \mathsf{x}, 1\geq p(x) \geq 0$
    - $\sum{p(x)} = 1$

### 3.3.2 Continuous Variable, Probability Density Function

- 연속확률변수에선 Probability Density Function 사용
- PDF의 조건:
    - P의 domain(정의역)은 x의 모든 가능한 상태의 집합이어야 함.
    - $\forall x \in \mathsf{x}, p(x) \geq 0$
    - $\int{p(x)dx} = 1$
- 확률분포 $u(x;a, b)$에서 x는 함수의 argument이고, **;**기호는  매개변수가 이후에 서술됨을 뜻함.(parameterization)

## 3.4 Marginal Probability(주변 확률)

- marginal probability distribution은 어떤 변수들의 집합에 관한 확률분포 정보를 알고있을 때 그 집합의 한 부분집합에 관한 확률분포.

![_config.yml]({{ site.baseurl }}/assets/ch3/1.png)

x에 대한 marginal distribution

- 물론 연속 변수에 대해선 sigma가 아니라 integral을 붙여야겠지요

## 3.5 Conditional Probability

- 어떤 사건이 발생했을 때, 다른 한 사건이 발생할 확률.
- $\mathsf{x}=x일 때,  \mathsf{y}=y$일 조건부 확률은 $p(\mathsf{y}=y| \mathsf{x}=x)$와 같이 표기한다.
- 조건부 확률 계산식:

$$P(\mathsf{y}=y|\mathsf{x}=x) = {P(\mathsf{y}=y, \mathsf{x}=x) \over P(\mathsf{x}=x)}$$

## 3.6 The chain rule of conditional probability

- 많은 random variable들로 구성된 joint probability distribution들은 아래와 같은 conditional distribution으로 분해될 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch3/2.png)

- 이를 product/chain rule of probability이라고 한다. 아래 예시.

![_config.yml]({{ site.baseurl }}/assets/ch3/3.png)


## 3.7 Independence and Conditional Independence

- 두 확률변수 $X$와 $Y$는 다음의 조건을 만족하면 독립:

$$\forall x \in X, y \in Y, p(X=x, Y=y) = p(X=x)p(Y=y)$$

- 개인적으로는 아래의 식이 더 와닫는 것 같다 (당연히 $X$와 $Y$를 바꾼 식도 성립하며 각 식의 양변에 $p(X=x)$ 또는 $p(Y=y)$를 곱하면 위 식이 유도됨).

$$p(X=x|Y=y) = p(X=x)$$

- 두 확률변수 $X$와 $Y$가 독립은 아니지만 또 다른 확률변수 $Z$에 대하여 조건부 독립인 경우도 있다.

$$\forall x \in X, y \in Y, z \in Z,  p(X=x, Y=y|Z=z) = p(X=x|Z=z)p(Y=y|Z=z)$$

- 조건부 독립의 정의 역시 풀어서 쓰는것이 더 이해하기 편한 듯 (아래 식의 양변에 $p(Y=y, Z=z)$를 곱한 후 정리하면 위 식이 유도됨).

$$p(X=x|Y=y,Z=z) = p(X=x|Z=z)$$

## 3.8 Expectation, Variance and Covariance

- Expectation
    - $x$ is discrite: $\mathbb{E}_{x \sim P}[f(x)] = \sum_x P(x)f(x)$
    - $x$ is continuous:  $\mathbb{E}_{x \sim P}[f(x)] = \int P(x)f(x)dx$
    - Expectation은 선형이다: $\mathbb{E}[\alpha f(x) + \beta g(x)] = \alpha \mathbb{E}[f(x)] + \beta \mathbb{E}[g(x)]$
- Variance: $\text{Var}(f(x)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2]$
- Covariance: $\text{Cov}(f(x), g(y)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])(g(y) - \mathbb{E}[g(y)])]$
    - Covariance는 두 변량이 얼마나 함께 변화하는지 나타내는 값이다.
    - 같은 변량 사이의 covariance는 분산.
    - Covariance 값은 두 변량의 스케일에 영향을 받는다. 이를 normalize 한 것이 correlation 이다.

## 3.9 Common Probability Distributions

자주 등장하는 분포들을 소개한다.

### 3.9.1 Bernoulli Distribution

- ~~양쪽 면이 나올 확률이 다를수도 있는~~동전던지기.

$$P(X=0) = \phi \\ P(X=1) = 1 - \phi \\ P(X=x) = \phi ^ x (1 - \phi)^{1-x } \\ \mathbb{E}[x] = \phi \\ \text{Var}(x) = \phi(1-\phi)$$

### 3.9.2 Multinoulli Distribution

- Categorical distribution이라고도 불린다.
- 분류를 수행하는 신경망의 마지막 층에서 softmax 함수를 쓰는 이유가 신경망을 이용해 이 분포를 표현하기 위함이다.
- Bernoulli distribution에서 확률 변수가 둘 중 하나의 값을 가지는것과 달리 multinoulli distribution에서는 $k$개의 값 중 하나를 가진다.
- $\vec{p} = \{p_1, p_2, ..., p_{k-1}\} =[0, 1]^{k-1}$로 parameterize 된다. $k$번째 확률값은 $1-\vec{1}^T\vec{p}$로 정해진다.

### 3.9.3 Gaussian Distribution

- 대망의 정규분포. Normal distribution이라고도 불린다.

$$N(x;\mu, \sigma^2) = \sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\pi\sigma^2}(x-\mu)^2)$$

- $\sigma^2$의 역수를 precision이라 부르고 $\beta$라고 쓰는데, 위 식을 $\sigma$ 대신 $\beta$를 이용하여 적는 경우도 많다.
- Gaussian distribution의 확률변수 $x$가 2차원 이상인 경우를 multivariate normal distribution이라고 하는데, 다음과 같이 표현한다.

$$N(\vec{x};\vec{\mu}, \Sigma) = \sqrt{\frac{1}{(2\pi)^n\det(\Sigma)}}\exp(-\frac{1}{2}(\vec{x}-\vec{\mu})^T \Sigma^{-1}(\vec{x}-\vec{\mu}))$$

- $\Sigma$는 모든 $\vec{x}$의 페어에 대하여 covariance를 계산한 covariance matrix이다.

### 3.9.4 Exponential and Laplace Distribution

- 딥러닝을 모델을 다룰 때 굉장히 뾰족한 분포를 표현해야 할 경우가 있다. 그럴 경우에 exponential distribution이 유용하다(~~아니 책에 이런식으로 설레개 써놓을거면 왜 유용한지도 좀 같이 적어놓지...~~). 왜 이런게 필요한지는 아래에서 설명하였다.

$$p(x;\lambda)=\lambda\text{I}_{x\ge0}\exp(-\lambda x)$$

- $\text{I}$는 indicator function이다. 아래첨자로 적힌 조건을 만족한 경우만 1이고 그렇지 않으면 0이다.
- Exponential distribution은 parameter $\lambda$를 조절하여 뾰족한 정도를 조절한다.

![_config.yml]({{ site.baseurl }}/assets/ch3/exp_dist.png)

- Exponential distribution을 조금 더 확장시켜 peak의 위치를 parameter $\mu$로 조절하게 한 것이 Laplace distrubiton이다.

$$\text{Laplace}(x;\mu, \gamma) = \frac{1}{2\gamma}\exp(-\frac{|x-\mu|}{\gamma})$$

![_config.yml]({{ site.baseurl }}/assets/ch3/laplace_dist.png)

- 이제 딥러닝에서 이러한 함수들이 왜 중요한지 이야기해 보자.
    - 딥러닝 모델을 학습할 때 과적합을 방지하기 위해 regularization을 사용하는 경우가 많다. 흔히 사용하는 regularization으로는 $L_1$regularization과 $L_2$ regularization(요즘에는 weight decay라고 부르는 경우가 더 많은 듯)이 있다. $L_1$ regularization의 경우 모델 weight들의 절대값의 합을, $L_2$ regularization의 경우 모델 weight들의 제곱의 합을 loss에 추가한다.
    - 각각의 regularization을 이용하여 학습한 모델은 학습된 weight의 분포에 차이가 생긴다. $L_1$regularization의 경우 0인 weight 값이 굉장히 많은 방향(sparse model)으로 학습되고 $L_2$regularization의 경우 적당히 작은 모든 weight값들이 적당히 작은 값을 유지한 채 학습된다. 왜 이렇게 학습되는지는 $L_1$, $L_2$ regularization loss의 함수 꼴을 생각하고 이 값을 줄이려면 모델 입장에서 weight값을 어떻게 조절하는 것이 이득인지 생각해보자.
    - 이제 베이지안 관점에서 두 regularization을 생각해보자. 베이지안 관점에서 $L_1$regularization은 weight의 prior distribution이 Laplace distribution인 경우, $L_2$regularization은 weight의 prior distribution이 Gaussian distribution인 경우이다. 따라서 더 뾰족한 Laplace distribution을 사전분포로 가정한 경우에 더 sparse한 모델이 학습되는 것이다.
    - 이 밖에도 딥러닝에서 굉장히 뾰족한 분포가 중요하게 이용되는 경우가 있을 것 같은데... 더이상은 잘 모르겠다.

### 3.9.5 The Dirac Distribution and Empirical Distribution

- ~~물리충이라면 익숙한~~델타펑션과 그것의 일반화인 empirical distribution이다 (아래 식은 순서대로 Dirac distribution, empirical distribution).

$$p(x)= \delta(x-\mu) \\ \hat{p}(\vec{x}) = \frac{1}{m} \sum_{i=1}^m\delta(\vec{x}-\vec{x}^{(i)})$$

- Empirical distribution에서 $\vec{x}^{(i)}$는 $i$번째 데이터라고 생각하면 편하다. 그래서 이름이 empirical distribution이다. 이 분포는 $n$차원 공간에 까시가 $m$개 있는 형태인데, 우리가 샘플링한 $m$개의 $n$차원 데이터를 표현했다고 생각하면 된다.

### 3.9.6 Mixtures of Distributions

- 종종 여러 분포의가 섞여있는 것을 표현해야 할 경우가 있다. 이럴 경우 아래처럼 표현할 수 있다.

$$P(x) = \sum_i P(c=i)P(x|c=i)$$

- $P(c=i)$는 multinoulli distribution인데, 이를 이용하여 여러 분포들 ($P(x|c=i)$)를 가중합 했다고 생각하면 된다.
- **3.9.5**에 등장한 empirical distribution 역시 혼합분포이다.
- 자주 쓰이는 혼합분포에는 mixtures of Gaussian이 있다. 데이터가 여러개의 Gaussian 분포로부터 샘플링 된 경우 mixtures of Gaussian이 된다.

## 3.10 Useful Properties of Common Functions

- 앞으로 자주 등장할 함수와 유용한 성질을 소개한다.
- Sigmoid
    - 딥러닝에서 Bernoulli distribution을 표현하기 위해 많이 쓰인다. 따라서 이진 분류 모델을 만들 때 sigmoid를 활성함수로 사용한다.
    - 자매품으로 multinoulli distribution을 표현하기 위해 많이 쓰이는 softmax 함수도 있다.
- Softplus
    - $\zeta(x) = \log(1+\exp(x))$
    - 이후 에 등장할 ReLU 함수 ($\max(0, x)$)를 0 근처에서 스무스하게 만든 함수라 생각하면 된다.
    - 함수값이 $(0, \infty)$의 범위를 가지기 때문에 모델을 이용해 Gaussian의 분산등을 표현하기에 적합하다.
- 알아두면 유용한 식들
    - $\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))$
    - $1 - \sigma(x) = \sigma(-x)$
    - $\log \sigma(x) = -\zeta(-x)$
    - $\frac{d}{dx}\zeta(x) = \sigma(x)$
    - $\forall x \in (0, 1), \sigma^{-1}(x) = \log(\frac{x}{1-x})$
    - $\forall x > 0, \zeta^{-1}(x) = \log(\exp(x)-1)$
    - $\zeta(x) = \int_{-\infty}^x\sigma(y)dy$
        - 이거 좀 확장해서 생각해보면 ReLU는 indicator function을 위와 같이 적분한 값이다.
    - $\zeta(x) - \zeta(-x) = x$
