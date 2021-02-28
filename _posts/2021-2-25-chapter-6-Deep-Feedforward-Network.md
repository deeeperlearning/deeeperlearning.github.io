이 장에서는 심층 신경망의 형태와 학습 방법에 대하여 다룬다. 먼저 심층 신경망이 무엇인지 잠깐 소개한 후 구체적인 내용을 다룰 예정이다.

우선 심층 순방향 신경망의 형태적인 특징은 다음과 같다.

- 이 책에서 다루는 신경망은 여러 층으로 구성되어 있다. 일반적으로 각 층은 input 벡터를 output 벡터로 mapping하는 함수 정도로 생각하면 된다.
- 각 층을 $f^{(i)}(x)$라고 표현한다면 전체 신경망은 $f(x) = f^{(1)} \circ f^{(2)} \circ ... \circ f^{(i)}(x)$이다.
- 값이 순방향으로만 전파되기 때문에 이러한 이름이 붙었다. 값이 되먹임되는 매커니즘이 추가된 recurrent neural network도 있는데 이는 추후에 등장한다.
- 신경망의 목표는 주어진 데이터를 잘 표현하는 함수를 근사하는 것이다. 예를들어 지도학습의 경우 input $x$, 이에 대응되는 label $y$에 대하여 $ y \approx f{(x)}$ 인 $f$를 찾는것이
  목표이다.

심층 순방향 신경망에서는 비선형성이라는 요소도 굉장히 중요하다. 기존의 머신러닝 기법들에서도 선형 모델의 한계가 명확하기 때문에 비선형 함수를 최적화하려는 노력이 계속되어 왔다. 이러한 논리는 심층 순방향
신경망에서도 유의하며 이를 위해 각 층의 output에 대하여 비선형함수인 activiation function을 적용한다. 기존의 머신러닝 알고리즘들도 모델에 비선형성을 추가하기 위해 여러 방법들을 사용했는데
신경망에서 activation function을 이용하는 이유와 기존 방법들의 단점은 다음과 같다.

- (기존 모델) RBF 커널에 기초하여 무한차원 커널을 사용하는 방법: 커널의 차원이 충분히 높으면 데이터를 표현하기에 충분한 capacity를 가진다. 하지만 overfitting이 발생할 위험이 크다.
- (기존 모델) 특화된 비선형 함수를 사람이 설계하는 방법: 해당 분야에 대한 지식이 충분하다면 좋은 방법이다. 하지만 좋은 비선형 함수를 설계하기 위해 수많은 연구가 필요하다.
- (심층 순방향 신경망) 비선형 함수 자체를 학습: 신경망의 각 층이 다음 층으로 신호를 보내기 전에 activation function을 적용하는 경우 신경망의 각 층은 비선형함수를 매개변수화하여 표현한다. 즉,
  우리는 매개변수를 학습 알고리즘을 통해 최적화하고 비선형함수를 학습할 수 있다. 이러한 방법을 사용하면 신경망의 형태를 조절하여 overfitting을 방지할수도 있고, 신경망의 형태가 적절한 함수족을 표현할 수
  있도록 선택하기만 하면 좋은 성능을 낼 가능성이 있는 비선형 함수를 학습할 수 있다.

## 6.1 Example: Learning XOR

이 장에서는 심층 순방향 신경망의 구체적인 형태를 이해하기 위해 간단한 예제를 풀어볼 것이다. 아직 학습 알고리즘을 소개하지 않았기 때문에 XOR 문제에 대하여 신경망의 해를 직접 구해볼 것이다(구한다는 표현이 좀
어색한데, 문제가 너무 간단해서 적절한 매개변수값을 유추할 수 있다).

XOR은 두 이진수 $x_1, x_2$에 대한 연산이다. 두 숫자 중 하나가 1이면 1을 반환하고 그렇지 않으면 0을 반환한다. 즉, 우리가 신경망으로 근사해야 하는 함수는 input이 $(x_1, x_2) = (
0, 0) \text{ or } (1, 1)$인 경우 0을, $(x_1, x_2) = (0, 0) \text{ or } (1, 1)$인 경우 1을 반환해야 한다.

$$X = \{[x_1, x_2]^T | x_1 \in \{0, 1\}, x_2 \in \{0, 1\} \}$$

우선 가장 간단한 형태의 선형 모델을 XOR 근사에 사용해보고 선형 모델의 한계를 살펴보자. 사용한 선형 모델과 앞 장에서 소개한 MSE error는 다음과 같다.

$$\vec{x} = [x_1, x_2]^T \\ f(\vec{x};\vec{w}, b) = \vec{x}^T\vec{w} + b \\ J(\vec{w}, b) = \frac{1}{4}\sum_{\vec{x} \in
X} (y - f(\vec{x};\vec{w}, b))$$

이전 장에서 MSE error에 대하여 선형 모델의 닫힌 형식의 표준방정식을 유도했는데, 이를 적용하면 $\vec{w} = \vec{0}, b = \frac{1}{2}$이다. 즉, 선형 모델은 모든 input에
대하여 0.5를 출력한다. 이는 선형 모델이 비선형 모델인 XOR를 표현할 수 없기 때문이며 아래의 그림을 보면 선형 모델이 이 문제를 풀지 못한 이유를 알 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch6/input_space.png)

이번에는 모델에 비선형성을 추가하기 위해 두 층으로 이루어진 선형 신경망을 이용해보자. 사용할 신경망의 계산은 다음과 같이 이루어진다.

$$\vec{h} = f^{(1)}(\vec{x}; W, \vec{c}) = g(W^T\vec{x} + \vec{c}) \\ y' = f^{(2)}(\vec{h}; \vec{w}) =
\vec{w}^T\vec{h}$$

식에서 $g$는 벡터의 각 성분별로 적용되는 ReLU 함수이다. 식의 $\vec{h}$는 신경망의 첫번째 층이 출력한 값, $y'$은 두번째 층이 출력한 값이라고 생각하면 된다. 신경망의 매개변수들을 다음과 같이
선택하면 신경망이 XOR 연산을 할 수 있게 된다.

$$W =\left[ \begin{matrix} 1 & 2 \\ 3 & 4 \\ \end{matrix} \right] \\ \vec{c} = [0, -1] \\ \vec{w} = [1, -2]$$

이제 두개의 층으로 이루어진 순방향 신경망이 XOR을 근사할 수 있었던 이유를 살펴보자. 아래는 신경망의 첫번째 층에서 데이터의 분포이다. 비선형 함수를 이용하여 input space가 왜곡되었고 결과적으로 선형
함수인 신경망의 두번째 층으로 근사할 수 있는 분포를 만들어냈다.

![_config.yml]({{ site.baseurl }}/assets/ch6/latent_space.png)

## 6.2  gradient based learning

neural network를 designing, training하는 것은 다른 ML model을 gadient descent(GD)로 training하는 것과 크게 다르지 않다. 가장 큰 차이는, Neural
network는 nonlinear이기 때문에 loss function이 non-convex여야한다는 점이다. SVM 등에서 사용하는 전역 수렴을 보장하는 convex optimization이 아닌, cost
function을 낮은 값으로 이끄는 역할만 하는 iterative gradient descent만 사용한다.

### 6.2.1 cost function

parametric model은 하나의 분포 $p(y|x;\theta)$를 정의. MLE를 이용해 learning을 수행. 일반적으로 cross entropy를 cost function으로 사용한다.

신경망에서 쓰이는 cost function에는 기본적인 cost function + regularization term이 들어간 경우가 많다.

#### 6.2.1.1 MLE를 이용한 Conditional Distributions 학습

대부분의 현대 신경망은 maximum likelihood를 이용해 훈련한다. 이 뜻은 cost function이 negative log-likelihood라는 뜻이다. 이 negative log likelihood는
cross entropy로도 서술 가능하다.

$$j(\theta) = -E_{x,y \sim \hat{p}_{data}}log \ p_{data}(y|x)$$

cost function의 구체적인 형태는 모형마다 다르며 $log\ p_{model}$의 구체적인 형태에 의존한다. 예를 들어 모델의 확률분포가 아래와 같다면,

$$p_{model}(y|x) = \mathcal{N}(y;f(x;\theta), I)$$

cost function은 MSE가 된다.

신경망 설계에서 cost function의 형태는 학습에 큰 영향을 미친다. cost funcion의 기울기는 학습 알고리즘이 잘 지도되도록 크고 예측 가능해야하며, 평평한 함수들은 사용되기 어렵다.

#### 6.2.1.2 Learning Conditional Statistics

Model이 전체 확률분포 $p(y\given x;\theta)$를 배우는 것이 아니라 $x$가 주어졌을 때의 $y$의 한 conditional statistics만 배우면 되 되는 경우가 있다. 변분법으로 아래 최적화 문제를 풀면,

![_config.yml]({{ site.baseurl }}/assets/ch6/fred0.png)

이런 최적화문제를 풀 때, 아래와 같은 결과가 나온다. 즉 $x$의 각 값에 대해 $y$의 평균을 예측한다는 의미가 된다.

![_config.yml]({{ site.baseurl }}/assets/ch6/fred1.png)

얻을 수 있는 통계량은 비용함수의 종류에 따라 다르다.

![_config.yml]({{ site.baseurl }}/assets/ch6/fred2.png)

위와 같은 최적화문제에서는 median을 예측하는 최적화 문제가 된다. 위 두 예시, 평균제곱오차나 평균절대오차는 기울기 기반 최적화에서 성능이 나쁠 때가 많다. 때때로 기울기가 saturate되기 때문이다.

### 6.2.2 Output Units

![_config.yml]({{ site.baseurl }}/assets/ch6/fred3.png)

output unit은 output layer을 뜻한다. cost function의 선택은 output unit의 선택과 밀접하게 연관되어 있다. MLE의 관점에서 output distribution의
negative log likelihood를 cost function으로 사용한다.

#### 6.2.2.1 Linear unit for gaussian output distributions

Linear unit은 Affine transformation에 기초한 output이며 nonlinear 함수를 거치지 않는다. linear unit은 conditional gaussian distribution의
평균을 구할 때 흔히 사용한다.

$$p(y|x) = \mathcal{N}(y;\hat{y}, I)$$

이 경우 5장에서 다루었듯 MSE가 cost function으로 쓰인다. linear unit을 사용하면 gaussian distribution의 covariance도 쉽게 학습할 수 있다.

#### 6.2.2.2 Sigmoid Units for Bernoulli Output Distributions

binary variable $y$를 예측하는 과제들이 많다. 이런 문제들을 MLE로 접근하고자 할 때에는 $P(y=1|x)$만 구하면 된다. 이런 과제에서는 output unit으로 S자형인 sigmoid 함수를
주로 사용한다.

$$\hat{y} = \sigma(w^Th+b)$$

이런 S자 함수가 필요한 이유는, 혹은 이 함수가 도출되는 이유는 아래와 같다. 정규화되지 않은(확률합이 1이 아닌) 확률분포를 정규화하려면 그 분포를 분포의 크기로 나누면 된다. 어떤 정규화되지 않은 로그 확률들이
y, z에서 선형이라고 할 때, y, z를 곱한 확률분포를 아래와 같이 나타낼 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch6/fred4.png)

즉, log likelihood의 정규화과정에서 sigmoid가 도출된다.

정규화에 기초한 확률분포들은 통계적 모형화를 다룰 때 자주 등장하며 그런 확률분포를 정의하는 $z$ 변수를 logit이라고 한다.

#### 6.2.2.3 Softmax Units for Multinoulli Output Distributions

가능한 값이 n가지 이상인 discrete variable을 표현해야할 때엔 softmax 함수를 사용하면 된다. (onehot encoded multilabel을 생각해보자) 이는 sigmoid같은 s자 함수의 한
일반화이다. 6.2.2.2에서 log likelyhood에 대한 gradient descent에 잘 부합해야 하므로 $P(y=1|x)$ 대신 $$z = log P(y=1|x)$$을 사용했다. Multinoulli에서
이를 사용하기 위해선 아래와 같은 확률분포와 그 예측 벡터 $\hat{y}$를 다루어야 한다.

$$\hat{y}_i = P(y=i|x)$$

벡터 $\hat{y}$가 유효한 확률 분포를 나타내려면, 각 성분 $\hat{y}_i$가 0과 1 사이의 값이어야 하며, 모든 성분의 합이 1이어야 한다. 먼저 선형 출력층은 아래와 같이 정규화되지 않은 로그 확률을
예측한다. 선형 출력층:

$$z = W^Th + b$$

그리고 이를 softmax에 넣어 원하는 분포 $\hat{y}$를 얻는다.

$$softmax(z)_i = {exp(z_i) \over \sum_jexp(z_j)}$$

exp 함수의 효과는 maximum likelihood를 이용해 softmax의 output이 y를 향하도록 하는 것이다. log likelihood가 softmax의 exponential을 상쇄하는 효과를 낸다는
점에서 아래와 같이 log softmax를 서술하는 것이 자연스럽다.

$$log \ softmax(z)_i = z_i - log\sum_jexp(z_j).$$

log likelihood 이외의 object function은 softmax함수와 잘 맞지 않을 때가 많다. log가 없는 object function을 사용하면, exp의 인수가 아주 큰 음수일 때 너무 작은
기울기가 나와 적절한 학습이 이루어지지 않을 수 있다.

softmax 함수의 input z에 특정 값을 더하거나 빼도 같은 값이 나오므로(exponential 나누기) max($z_i$)를 $z$에서 빼서 overflow가능성이 적은 softmax 계산을 할 수 있다.

신경과학의 관점에서 softmax는 그에 관여하는 단위들 사이의 일종의 경쟁이라고 할 수 있다. (softmax의 합은 항상 1) 사실 softmax는 max보다는 argmax에 가깝다고 할 수도 있다. (즉 몇번째
i의 가능성이 가장 큰가!)

soft는 미분 가능 함수라는 사실에서 비롯된 이름이다.

#### 6.2.2.4 Other Output Types

위에서 한 논의들을 정리하자면, 신경망을 위해 조건부 확률분포 $p(y|x;\theta)$를 정의했을 때, maximum likelihood 원리가 제시하는 cost function은 $-log\ p(
y|x;\theta)$이다.

신경망을 함수 $f(x;\theta) = w$라고 한다면, 결과적으로 손실함수를 $-log \ p(y;w(x))$로 해석할 수도 있다.

x가 주어졌을 때 y에 대한 gaussian distribution을 배운다고 할 때 분산의 추정량은 분산을 계산하는 식으로도 도출할 수 있다(제평-평제) 하지만 특수 경우를 위한 코드 작성이 필요없는 일반화된
접근법은 분산을 $w=f(x;\theta)$로 제어되는 분포 $p(y|x)$의 한 속성으로 두는 것이다. 이 경우 cost function은 log likelyhood로 구할 수 있다.

대각행렬보다 더 복잡한 covariance matrix나 pricision matrix를 학습해야하는 경우는 드물다. covariance matrix이 완전하고 조건부일 때는 positive definite한
매개변수를 선택해야 한다. 이런 계산에서 문제는 역행렬을 구해야 하는데 $O(d^3)$이라 너무 비싸다.

x의 한 값에 대해 y 공간에 여러 개의 peak이 존재하는 조건부 분포 $p(y|x)$로부터 real value들을 예측해야하는 경우가 있다. 이를 다봉분포 회귀(multimodal regression)이라고
한다. 이 경우 출력을 표현할 때 gaussian mixture distribution을 사용하는 것이 좋다.

![_config.yml]({{ site.baseurl }}/assets/ch6/fred5.png)

gaussian mixture distribution을 출력으로 사용하는 신경망을 mixture density network라고 한다. conditional Gaussian mixtures를 output으로 사용하는
경우 수치적 불안정성으로 결과가 그리 좋지 못하다는 연구들이 있는데 gradient clipping이라는 방법으로 해결할 수 있다. 또 scale the gradients heuristically하는 방법도 있다.

Gaussian mixture output은 speech geneative model에 효과적이며, movements of physical objects 모델에도 사용된다.

## 6.3 Hidden Units

- Hidden units에 대해서는 절대적인 이론적 가이드라인이 있는게 아니라, 다양한 설계가 연구중임
- Rectified linear unit (ReLU)가 기본적인 hidden unit으로 널리 쓰임
- 소개할 몇몇 hidden unit은 미분 불가능한 지점을 포함하기도 함
    - Ex) $g(z)=max(0,z)$ 는 $z=0$에서 미분 불가능함
    - 그러나 이러한 경우가 많지는 않으며, 소프트웨어에서 인위적으로 특정 값을 배정하여 수치연산 오류를 방지하곤 함

### 6.3.1 Rectiﬁed Linear Units and Their Generalizations

- Actiation function: $g(z)=max(0,z)$
    - Linear unit과 유사하기 때문에 최적화 하기 용이함 (차이점: 절반의 결과값이 0이 됨)
    - 미분값은 active unit의 경우 상수로 일정하고, 이차미분값은 모두 0임
- 일반적으로 affine matrix를 input으로 이용: $h=g(W^⊤x+b)$
    - 보통 $b=0.1$과 같이 작은 양수로 초기값을 설정 $\rightarrow$ 대부분의 unit이 초기 상태부터 active하도록 함

![_config.yml]({{ site.baseurl }}/assets/ch6/ReLU.png)
https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e

- ReLU 보다 성능이 높도록 개량된 버전의 예시
    - $h_i=g(z,\alpha)_i=max(0,z_i)+\alpha_imin(0,z_i)$ 에서,
        - $\alpha_i=-1$ : 절대값을 계산하는 absolute value rectification, $g(z)=\|z\|$
        - $\alpha_i$가 0.01과 같이 작은 값으로 고정: leaky ReLU
        - $\alpha_i$가 학습 가능한 parameter: parametric ReLU, PReLU
        - Maxout unit: ReLU를 좀 더 변형 $\rightarrow$ 여러 input 중 가장 큰 값을 내뱉음

![_config.yml]({{ site.baseurl }}/assets/ch6/maxout.PNG)
http://www.simon-hohberg.de/blog/2015-07-19-maxout

### 6.3.2 Logistic Sigmoid and Hyperbolic Tangent

- ReLU 이전에는 대부분 sigmoid activation function, hyperbolic tangent activation function을 사용함
    - Sigmoid: $g(z)=\sigma(z)$
    - Hyperbolic tangent: $g(z)=tanh(z)$
    - $tanh(z)=2\sigma(2z)-1$ 이기 때문에 두 함수는 유사하게 작동함
    - 이 둘은 saturation되어 hideen unit으로 잘 사용되지는 않음
    - 다만, saturation을 상쇄해주는 cost function이 함께 있을 때, output 함수로 종종 사용되기도 함

![_config.yml]({{ site.baseurl }}/assets/ch6/sigmoid.png)
https://www.researchgate.net/figure/The-sigmoid-and-hyperbolic-tangent-activation-functions_fig2_265486784

- Sigmoid 보다는 hyperbolic tangent를 사용하는 것이 더 나음
    - $\sigma(0)=0.5$이지만, $tanh(0)=0$이기 때문에, 0 근처에서 y=x와 더 가깝게 작용하여 더 간단하게 학습하는 모델이 됨
    - Recurrent network, probabilistic model, autoencoder 등에서 특정 제한 조건이 있는 경우, saturation이라는 단점에도 불구하고 sigmoid function을
      사용하기도 함
        - 예) Firing rate의 상한이 있는 뉴런의 활동을 모사

### 6.3.3 Other Hidden Units

- 다른 hidden unit들도 다양하지만, 잘 쓰이지는 않음
    - Cosine function을 이용했을 때 MNIST 벤치마크 데이터셋에서 테스트 성능이 1퍼센트 향상됨
    - Softmax는 output으로는 흔히 쓰이지만 hidden unit으로는 잘 쓰이지 않음  (10장에서 다룰 예정)
    - radial basis function (RBF), Softplus, Hard tanh 등 최근까지도 다양한 함수들이 제시되며 활발히 연구되고 있는 분야임

## 6.4 Architecture Design

- Neural network은 여러 개의 레이어로 구성되어 있는데, 이를 어떤 너비, 얼마만큼의 깊이로 쌓을지 결정해야 함
- 각 레이어는 이전 레이어의 출력이 다음 레이어의 입력이 되는 chain-based 구조를 가짐

### 6.4.1 Universal Approximation Properties and Depth

- Feature과 output을 행렬곱으로 맵핑하는 linear model은 convex optimization 을 사용하여 해결할 수 있지만, 우리가 학습하려는 대상은 nonlinear한 함수일 수 있음

- Universal approximation theorem: Hidden layer가 있는 feedforward network는 어떠한 함수든 근사할 수 있음 (nonlinear 함수 포함)

- MLP가 함수를 표현할 수 있다고 해도 학습이 실패할 수 있는 두 가지 이유가 있음
    - 최적화 알고리즘이 학습해야할 함수에 적합하지 않을 수 있음
    - 학습 알고리즘이 overfitting 되어 엉뚱한 함수를 학습할 수 있음

- "공짜 점심은 없다" 이론에서 언급했듯이, 모든 것을 다 만족시키는 만능 AI란 없음
    - Universal approximation theorem이란 주어진 함수를 MLP를 이용해 표현 가능하다는 거지, 학습한 함수가 test set에 대해서도 만능으로 잘 작동한다는 뜻은 아님

- 따라서 parameter 수를 줄여가며 원하는 함수를 representation 해야 함
    - 아래 그림) 레이어를 깊게 쌓아감에 따라 decision boundary가 단순해지는 에시
        - Activation function은 절댓값을 사용하였기 때문에 레이어 하나를 지날 때마다 반절씩 접는 것으로 표현함
        - Affine transformation은 반절 접는 위치를 정해줌
        - 레이어를 지날수록 근사해야하는 함수가 단순해지고, 그렇기 때문에 generalization error가 줄어든다고 볼 수 있음

![_config.yml]({{ site.baseurl }}/assets/ch6/Fig6.5.PNG)

- 어느 지점까지는 network의 depth, size가 확보되어야 하지만, 무조건 크다고 성능이 향상되는 것은 아님 (아래 두 그림 예시)

![_config.yml]({{ site.baseurl }}/assets/ch6/Fig6.6.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch6/Fig6.7.PNG)

### 6.4.2 Other Architectural Considerations

- 여기까지는 neural network를 단순히 연결로만 보고 레이어의 깊이와 너비(유닛 수)에 대해서 주로 논했지만, 사실은 목적에 따라 더 다양함
    - Convolution network (9장)
    - Sequence 데이터를 처리하기 위해 만들어진 recurrent network
- 기본적으로 main chain을 어떻게 설계하는지가 중요하지만, 이 외에도 skip connection, attention, convolutional filter 등의 설계 역시 중요함
    - Skip connection: $i$번째 레이어에서 $i+1$번째가 아닌 $i+2$번째 레이어로 바로 넘겨서 gradient flow를 용이하게 만드는 기법
    - Attention: Representation 의 일부만 선택적으로 넘겨서 효율을 향상시키는 방법
    - Convolutional network: 도메인 내에서 filter를 공유함으로써 적은 파라미터로도 복잡한 패턴을 효율적으로 잡아낼 수 있게 설계된 방법

## 6.5 Back-Propagation and Other Differentiation Algorithms

- Forward propagation : Input $\boldsymbol x$ 의 정보를 각 layer의 hidden unit에 전달하여 output $\hat{\boldsymbol{y}}$ 를 계산하는 과정.
- Back-propagation : Forward propagation의 반대방향으로 gradient를 계산하는 과정

### 6.5.1 Computational Graphs

각각의 변수를 하나의 node로 하여 계산과정을 graph 형태로 나타내는 방법.

![_config.yml]({{ site.baseurl }}/assets/ch6/computational graph.png)

위의 그림은 weight decay penalty가 포함된 선형회귀 모형을 computational graph 방식으로 나타낸 그림이다. 즉, $\hat{\boldsymbol y} = \boldsymbol
x^{\top} \boldsymbol w$와 $\lambda \sum_i{w_i^2}$ 를 계산하는 과정.

### 6.5.2 Chain Rule of Calculus

Back-propagation은 chain rule을 이용하여 gradient를 효율적으로 계산하는 알고리즘이다.

$\boldsymbol x \in \mathbb{R}^m,\ \boldsymbol y \in \mathbb{R}^n,\ z \in \mathbb{R}$ 이고 $\boldsymbol y = f(\boldsymbol
x),\ z = g(\boldsymbol y)$ 일 때, chain rule은 아래와 같이 적힌다.

$$\frac{\partial z}{\partial x_i} = \sum_{j}{\frac{\partial z}{\partial y_j}}\frac{\partial y_j}{\partial x_i}\ \ or\ \
\nabla_{\boldsymbol x}z = \left(\frac{\partial\boldsymbol y}{\partial\boldsymbol x}\right)^{\top}\nabla _{\boldsymbol
y}z$$

### 6.5.3 Recursively Applying the Chain Rule to Obtain Back-propagation

- 역전파를 하는 과정에서 chain rule을 적용할 때 동일한 계산을 여러번 하게됨.

  ![_config.yml]({{ site.baseurl }}/assets/ch6/recursive1.png)

- 따라서 역전파의 방향을 따라 각 노드의 gradient 값을 저장하면 반복적인 계산을 피할 수 있다.

  ![_config.yml]({{ site.baseurl }}/assets/ch6/recursive2.png)

- 하지만 gradient를 저장할 메모리가 증가한다는 단점이 있다.

### 6.5.4 Back-Propagation Computation in Fully-Connected MLP

- Forward propagation

![_config.yml]({{ site.baseurl }}/assets/ch6/algo6_3.png)

- Back-propagation
- ![_config.yml]({{ site.baseurl }}/assets/ch6/algo6_4.png)

### 6.5.5 Symbol-to-Symbol Derivatives

- Symbolic representation : 대수 표현과 computational graph와 같이 symbol을 이용하여 계산과정 또는 식을 표현
- 역전파를 진행할 때 computational graph와 입력 데이터의 실제 값(numerical value)을 이용하여 gradient의 실제 값을 도출하는 접근법을 "symbol-to-number"라고 한다.

    - Torch
- Computational graph에 추가적인 노드를 만들고 필요한 미분을 symbolic하게 저장하는 방식을 "symbol-to-symbol"이라고 한다.
    - TensorFlow

  ![_config.yml]({{ site.baseurl }}/assets/ch6/sym2sym.png)

    - 이러한 미분들은 하나의 computational graph를 만들기 때문에 또 다시 역전파를 진행할 수 있고, 즉, higher-order의 미분을 얻을 수 있다.
- "symbol-to-symbol"이 "symbol-to-number"보다 넓은 개념.

### 6.5.6 General Back-Propagation

$\mathcal{G}$ : computational graph.

$\mathsf{V}$ : variable(텐서) ($\mathcal G$에서 노드에 해당) .

get_operation($\mathsf V$) : $\mathsf V$를 계산하는 오퍼레이터를 리턴.

get_consumer($\mathsf V, \mathcal G$) : $\mathsf V$의 자식 변수를 리턴.

get_inputs($\mathsf V, \mathcal G$) : $\mathsf V$의 부모 변수를 리턴.

op.bprop($\text{inputs},\mathsf X, \mathsf G$) : $\sum_i(\nabla_{\mathsf X}\text{op.f(inputs)}_i)\mathsf G_i$ (chain
rule)

- 역전파 알고리즘

  ![_config.yml]({{ site.baseurl }}/assets/ch6/algo6_5.png)

- build_grad 알고리즘

  ![_config.yml]({{ site.baseurl }}/assets/ch6/algo6_6.png)

- n개의 노드를 가진 computational graph는 directed acyclic graph이기 때문에 gradient를 계산하기 위한 cost는 최대 $O(n^2)$이다.

### 6.5.7 Example: Back-Propagation for MLP Training

아래와 같이 히든 레이어가 하나인 퍼셉트론 문제를 생각해보자.

![_config.yml]({{ site.baseurl }}/assets/ch6/MLP.png)

$$J = J_{MLE}+\lambda\left(\sum_{i,j}\left(W_{i,j}^{(1)}\right)^2+\sum_{i,j}\left(W_{i,j}^{(2)}\right)^2\right)$$

$\nabla_{\boldsymbol W^{(1)}}J$ 와 $\nabla_{\boldsymbol W^{(2)}}J$ 은 두가지 길(weight decay cost와 cross entropy cost)을 통해 계산할
수 있다.

- Weight decay cost

    - 단순히 $2\lambda \boldsymbol W^{(i)}$로 주어짐.
- Cross entropy cost
    - $\boldsymbol U^{(2)}$에 대한 gradient를 $\frac{\partial J_{MLE}}{\partial \boldsymbol U^{(2)}} = \boldsymbol G$라 하면
      $\boldsymbol W^{(2)}$에 대한 gradient는 아래와 같다.

  $$\nabla_{\boldsymbol W^{(2)}}J_{MLE} = \boldsymbol H^{\top} \boldsymbol G$$

    - $\boldsymbol U^{(1)}$에 대한 gradient를 $\frac{\partial J_{MLE}}{\partial \boldsymbol U^{(1)}} = \boldsymbol
      G^{\prime}$라 하면(relu가 있기 때문에 $\boldsymbol U^{(1)}$의 component 중 0 보다 작은 애들은 gradient를 0으로) $\boldsymbol W^{(1)}$에
      대한 gradient는 아래와 같다.

  $$\nabla_{\boldsymbol W^{(1)}} J_{MLE} = \boldsymbol X^{\top} \boldsymbol G^{\prime}$$

### 6.5.8 Complications

- 어떤 operator가 두개의 텐서를 리턴할 경우, 두개의 출력이 있는 단일 작업으로 구현하는 것이 가장 효율적이다.
- 역전파 과정에서 여러개의 텐서를 더하는 경우가 많은데, 각각의 텐서를 따로 계산한 후에 전체를 더하게 되면 메모리 병목현상이 발생함. 따라서 하나의 버퍼를 유지하면서 하나씩 더하는게 메모리 적약에 효율적이다.
- 역전파과정은 여러가지 데이터 타입을 다뤄야 하기 때문에 데이터 타입별 처리 방식을 잘 디자인 해야한다.
- gradient가 정의되지 않는 operator가 있을 경우 사용자가 알 수 있도록 tracking이 잘 되어야 한다.

### 6.5.9 Differentiation outside the Deep Learning Community

- 역전파 알고리즘은 자동미분의 접근법중 하나일 뿐이고, 다른 방법들도 존재한다.
- 자동미분에는 대표적으로 forward/reverse mode accumulation이 존재하는데 역전파 알고리즘은 reverse mode에 속함.
    - reverse mode accumulation (output에서 input 방향)

      $$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial w_1}\frac{\partial w_1}{\partial x} = \left(
      \frac{\partial y}{\partial w_2}\frac{\partial w_2}{\partial w_1}\right)\frac{\partial w_1}{\partial x}= \left(
      \left(\frac{\partial y}{\partial w_3}\frac{\partial w_3}{\partial w_2}\right)\frac{\partial w_2}{\partial
      w_1}\right)\frac{\partial w_1}{\partial x}$$

    - forward mode accumulation (input에서 output 방향)

      $$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial w_{n-1}}\frac{\partial w_{n-1}}{\partial x} =
      \frac{\partial y}{\partial w_{n-1}}\left(\frac{\partial w_{n-1}}{\partial w_{n-2}}\frac{\partial w_{n-2}}{\partial
      x}\right)= \frac{\partial y}{\partial w_{n-1}}\left(\frac{\partial w_{n-1}}{\partial w_{n-2}}\left(\frac{\partial
      w_{n-2}}{\partial w_{n-3}}\frac{\partial w_{n-3}}{\partial x}\right)\right)$$

- Chain rule을 적용할 때 미분의 순서를 최적화시키는 문제는 NP-complete 문제로 알려져있다.

    - TensorFlow나 Theano는 휴리스틱 알고리즘을 사용해서 computational graph를 간소화 하는 방식으로 cost를 줄인다고 합니다.
- 머신러닝 분야는 특이하게 특정 라이브러리를 사용해서 코드를 짜기 때문에 많은 제약이 따르는데, 역전파 알고리즘을 커스텀할 수 있다는 장점도 있다.

### 6.5.10 Higher-Order Derivatives

- 사실 high order 미분이 딥려닝에서 자주 쓰이진 않고, Hessian 행렬의 몇가지 특징들 때문에 가끔 사용된다.
- 하지만 Hessian 행렬을 직접 계산하는 것은 cost가 너무 크기 때문에 Krylov method를 사용해서 간접적으로 구한다.

  $$\boldsymbol H \boldsymbol v = \nabla_{\boldsymbol
  x}\left[(\nabla_{\boldsymbol x}f(\boldsymbol x))^{\top}\boldsymbol v\right]$$

    - outer gradient는 inner gradient에만 적용됨.

## 6.6 Historical Notes

이 장에서는 지금까지 딥러닝이 발전해 온 과정을 간략하게 소개한다. 역사를 모두 알 필요는 없지만 각 시대별로 어떠한 이유에 의해 모델이 변경되고 발전해왔는지 기억해두면 좋을 것 같다.

- 시대별 큼직한 사건들
    - 17세기: back-propagation의 근간이 되는 chain rule이 발명됨.
    - 1940년대: function approximation 기법들이 사용되었다. 주로 선형 모델들을 이용했는데, XOR문제를 해결하지 못하는 등의 한계로 사장되었다.
    - 1960, 1970년대: 비선형 함수를 표현하기 위해 여러 층으로 이루어진 신경망이 도입되었다. 이러한 신경망의 학습을 위해 back-propagation이 개발되었다.
    - 1990년대 ~ 2006년: 1990년대에 신경망 연구가 굉장히 활발했지만 다른 2006년가지는 다른 머신러닝 기법들이 더 많이 사용됨.
- 최근 딥러닝이 많이 사용되는 이유
    - 사실 신경망 알고리즘에 중요한 기술들은 1980년대 이후로 크게 변한게 없다. 그럼에도 크게 발전한 이유는 크게 두 가지 이유가 있다.
        - 굉장히 큰 데이터셋을 사용할 수 있게 되었다. 이에 따라 모델이 데이터에 대해 일반화 할 수 있는 수준이 올라갔다.
        - 컴퓨터의 연산 능령이 크게 향상되었고 소프트웨어 인프라가 좋아졌다.
    - 알고리즘 자체가 발전한 점도 몇가지 있다.
        - MSE loss 대신 cross-entropy 형태의 loss를 사용하기 시작했고 MLE 관점에서 접근하기 시작했다. 이는 sigmoid나 softmax output을 사용하는 모델들의 성능을 크게
          향상시켰다.
        - Sigmoid 형태의 활성화함수를 ReLU로 대체하였다. 이전에는 sigmoid 형태의 활성화함수가 작은 모델에서 더 좋은 성능을 내는 점, 미분 불가능한 함수를 사용하면 안된다는 믿음 때문에
          ReLU를 사용하지 않았다.
