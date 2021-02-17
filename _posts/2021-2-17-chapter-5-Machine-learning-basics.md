## 5.3 Hyperparameters and Validation Sets

- Hyperparameter: 모델이 학습하는게 아니라 사용자가 값을 설정하는 parameter로, 대부분의 머신러닝 알고리즘이 가지고 있음
  - 예) polynomial regression에서 입력 feature의 차수, weight decay에서 $\lambda$
- 최근 hyperparameter까지 최적화해주는 모델들이 나오기는 하지만, 일반적으로 적절한 값을 선택하는 것은 어려운 문제임
- 보통 여러 후보 값으로 조절해가며 모델의 성능을 테스트함 $\rightarrow$ validation set 이용
  - validation set: test set이 아닌 training set의 일부로, 학습할 때 사용하지 않음
  - 일반적인 데이터 분배 비율: training : validation : test = 6 : 2 : 2

## 5.3.1 Cross-Validation

- 데이터를 위의 세 그룹으로 나누어 알고리즘을 평가할 때, test set의 크기가 충분히 크지 않은 경우엔 통계적 불확실성으로 인해 제대로 평가하기 부족함
- 따라서 데이터를 임의의 여러 그룹으로 나누어 평가를 반복하는 방법임
- k-fold cross-validation: 데이터를 서로 겹치지 않게 k개의 그룹으로 나누고, 각각의 그룹을 test set으로 사용하는 총 k번의 평가를 함
![_config.yml]({{ site.baseurl }}/assets/ch5/k-fold.PNG)
![_config.yml]({{ site.baseurl }}/assets/ch5/monte carlo.PNG)

## Monte-Carlo Cross Validation

- 일정 비율로 test set을 임의로 뽑아 테스트 하는 과정을 반복함
- 같은 데이터가 여러번 test set에 등장할 수 있다는 점이 k-fold cross validation과의 가장 큰 차이임

![_config.yml]({{ site.baseurl }}/assets/ch5/cross difference.PNG)

## 5.4 Estimators, Bias and Variance

- 머신러닝에서 유용하게 사용되는 통계적인 도구들에 대한 소개

## Point Estimation

- 단 하나의 best 예측을 뽑기 위한 방법
- 대상은 하나의 파라미터일 수도, linear regression 같이 벡터 파라미터일수도 있음
- $\hat{\theta}$ : 파라미터 $\theta$의 estimator, true $\theta$에 가까울수록 좋은 estimator임

## Function Estimation

- 입력 벡터 $x$에 대한 $y$를 예측하는 경우, 해당 함수에 대한 estimation
- 함수 공간에서의 point estimation이라 할 수 있음

## Bias

- data들로부터 추정한 $\hat\theta_m$의 기댓값과 true $\theta$와의 차이
![_config.yml]({{ site.baseurl }}/assets/ch5/eq5_20.PNG)
- unbiased: $E(\hat\theta_m) = \theta$
- asymptotically unbiased: 데이터 갯수가 무한대로 가면 unbiased 해지는 경우

## Example: Bernoulli Distribution

- mean이 unbiased estimator가 됨

![_config.yml]({{ site.baseurl }}/assets/ch5/Bernoulli.PNG)

## Example: Gaussian Distribution Estimator of the Mean

- sample mean이 unbiased estimator가 됨

![_config.yml]({{ site.baseurl }}/assets/ch5/Gaussian.PNG)

## Example: Estimators of the Variance of a Gaussian Distribution

- sample variance는 true variance의 biased estimator임

![_config.yml]({{ site.baseurl }}/assets/ch5/sample variance.PNG)

- unbiased estimator를 만들기 위해서는 분모에 $m$ 대신 $(m-1)$을 사용해야 함

![_config.yml]({{ site.baseurl }}/assets/ch5/sample variance_m-1.PNG)

## 5.4.3 Variance and Standard Error
- Variance: 추산한 값이 데이터 샘플마다 얼마나 크게 달라지는지에 대한 지표, $Var(\hat\theta)$
- Standard error: variance의 루트 값, $SE(\hat\theta)$
- 낮은 bias, 낮은 variance를 가지는 estimator가 대체로 우리가 원하는 형태임
- 유한한 데이터로 통계적인 수치를 계산할 때는 항상 불확실성을 내포함
- 같은 distribution에서 얻어진 데이터들이라 하더라도, 통계치는 달라질 수 있고, 이 달라지는 정도를 정량화하기 위한 도구임
- 평균 $\hat\mu_m$의 standard error:
![_config.yml]({{ site.baseurl }}/assets/ch5/SE.PNG)
- 평균 $\mu$의 95% 신뢰 구간:
![_config.yml]({{ site.baseurl }}/assets/ch5/confidence.PNG)
  - 알고리즘 A의 error에 대한 95퍼센트 신뢰구간의 upper bound가 알고리즘 B의 error에 대한 95퍼센트 신뢰구간의 lower bound보다 작다면 알고리즘 A가 더 낫다고 하곤 함

## Example: Bernoulli Distribution
- Bernoulli distribution의 estimator로 평균 값을 가정할 때, 데이터 수 m이 증가할수록 estimator의 variance는 감소함
![_config.yml]({{ site.baseurl }}/assets/ch5/variance_bernoulli.PNG)

## 5.4.4 Trading oﬀ Bias and Variance to Minimize Mean Squared Error
- Bias와 variance는 estimator error의 서로 다른 원인임
  - Bias: 함수나 파라미터의 참 값에서 예상되는 편차를 측정함
  - Variance: 데이터 샘플링으로 인해 발생할 가능성이 있는 예측치 값과의 편차를 나타냄
  - 일반적으로 bias와 variance는 아래 그림과 같이 trade-off 관계를 가짐
  - 이 때 cross-validation을 이용해 bias와 variance의 변화를 살피는 방법이 널리 사용됨
![_config.yml]({{ site.baseurl }}/assets/ch5/optimal capacity.PNG)
  - 또는 bias와 variance를 모두 포함하는 mean squared error (MSE)를 최소화하는 조건을 탐색함
![_config.yml]({{ site.baseurl }}/assets/ch5/MSE.PNG)
  - 일반적으로 capacity가 증가하면 bias는 줄어들고 variance는 증가함

## 5.4.5 Consistency
- Training set의 크기가 증가하면 point estimates가 참값에 수렴하는 성질
![_config.yml]({{ site.baseurl }}/assets/ch5/consistency.PNG)
- Consistency가 성립한다면 데이터 수가 증가할 때 bias가 줄어듬을 보장함
- 하지만 역은 항상 성립하지는 않음 $\rightarrow$ bias가 줄어든다고 consistency가 성립하지는 않음
  - 예) Dataset에서 normal distribution의 평균을 추정할 때, 무조건 첫번째 샘플을 estimator로 사용한다면 unbiased이긴 하지만, 데이터 개수가 무한대가 될 때 unbiased 해지는 것은 아니므로 consistency라 할 수 없다.


## 5.9 Stochasitic Gradient Descent

좋은 일반화를 위해선 dataset의 크기가 커야하지만 동시에 dataset의 규모가 증가하면 계산비용이 커진다. 기계학습 알고리즘이 사용하는 cost function을 training examples of some per-example loss function으로 분해할 수 있다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36f5a770-6747-4846-bcd5-43c0586c5e83/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36f5a770-6747-4846-bcd5-43c0586c5e83/Untitled.png)

여기서 $L$은 per-example loss $L(x,y,\theta) = -log\ p(y|x;\theta)$. 

Gradient descent를 이용하기 위해선 알고리즘의 cost function $\nabla_{\theta} J(\theta)$를 계산해야 한다. 연산의 코스트는 데이터의 수에 대해 선형의 시간복잡도를 가진다. 

SGD의 아이디어는 gradient가 하나의 기댓값이라는 점에서 출발한다. SGD에서는 training set에서 작은 데이터 단위 minibatch $\mathbb{B}$를 뽑아 학습에 사용한다. minibatch의 사이즈는 $m'$이며, 이 minibatch로 gradient를 추정하면 아래와 같다. 

$$g = {1 \over m'} \nabla_{\theta}\sum_{i=1}^{m'}L(x^{(i)} y^{(i)}, \theta)$$

예전엔 non-convex함수 최적화에 GD를 사용하지 않아야한다는 인식이 많았으나, 지금은 다름. 

m의 크기만 고정된다면 training data의 크기가 아무리 커져도 학습 비용이 $O(1)$이 될 것...?

### 5.10 Building a Machine Learning Algorithm

Deep learning Algorithm들은 보통 다음의 recipe을 지닌다.:

- specification of a dataset
    - 예를 들어 Iris flower data set, X: n*m, y: m
- a cost function
    - $J(w, b) = something$
- an optimization procedure
    - ADAM
    - k-means clustering, decision tree같은 알고리즘은 non-parameter method라 GD를 못쓴다. 즉 다양한 optimization 방법들이 있음.
- a model
    - $p_{model}(y|x) = N(y;x^Tw+b, 1)$ → 선형회귀

## 5.11 Challenges Motivating Deep Learning

Linear regression 등 5장에서 설명한 알고리즘들은 speech recognition, object recognition 등의 문제들은 잘 풀지 못한다. 이번 자에서는 복잡한 함수들을 배우는데 swallow model들이 적합하지 않다는 점 등을 다룬다. 

### 5.11.1 The curse of dimensionality(차원의 저주)

기계학습에서 자료의 차원이 아주 높을 때 풀기 어려워질 때가 많다. 변수의 개수에 따라 서로 다른 구성/조합의 개수가 지수함수적으로 증가한다. 차원의 저주가 초래하는 어려움 중 하나는 statistical challenge이다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/35c6048d-720c-49a1-98c2-df00fb37d324/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/35c6048d-720c-49a1-98c2-df00fb37d324/Untitled.png)

그림 설명: 1차원에서는 변수가 하나 뿐이고, 구별해야 할 변수의 다른 값들은 10개뿐이다. 그런데 2, 3차원으로 가면 지수적으로 구별해야할 값들이 증가한다. 1차원에서와 비슷하게 공간의 구멍들에 데이터를 채우려면 데이터의 양도 지수적으로 증가해야 한다. 

위 그림에서 보듯, x의 가능한 구성들의 개수가 training example의 수보다 훨씬 크면 statistical challenge가 발생한다. 데이터량이 한정적인 상황에서 차원이 높아지면 data가 하나도 배치되지 못한 격자 칸도 많아질 것이다. 

이렇게 차원의 저주에 걸린 경우 input data에 대해 어떻게 추론해야 할까? 전통적인 기계학습 알고리즘들은 new input data와 가장 가까이에 있는 training data의 출력와 이 new intput의 출력이 같을 것이라 가정한다. 

### 5.11.2 Local Constancy and Smoothness Regularization

ML 알고리즘의 잘 일반회되기 위해 알고리즘이 배워야 할 함수의 종류에 대한 prior belief를 알고리즘에 제공해야 한다. 많이 쓰이는 암묵적 prior로는 smoothness prior(local constancy prior)이 있다. 이 prior은 "함수가 작은 영역 안에서 아주 크게 변해서는 안 된다"는 제약을 뜻한다. 간단한 구조의 알고리즘들은 좋은 일반화를 보장하기 위한 수단이 이 prior밖에 없다. 

- smoothness prior이 충분한 prior이 될 수 없는 이유

smoothness prior은 특정 조건을 만족하는 함수 $f^*$가 아래와 같은 성질을 띄는 것을 뜻한다.

$$f^*(x) \approx f^*(x+\epsilon)$$

이런 smoothness prior의 극단적인 예로 k-nearest neighbor가 있다. k = 1일 때, the number of distinguishable regions은 be more than the number of training examples을 넘을 수 없다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/826ebd76-dd5e-4d7a-bf65-461c2f49da6e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/826ebd76-dd5e-4d7a-bf65-461c2f49da6e/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4693ad38-3664-41d0-8a32-587e705c5b1c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4693ad38-3664-41d0-8a32-587e705c5b1c/Untitled.png)

Decision tree 방법에도 smoothness만 의존하는 학습의 한계들이 적용된다. leaf node가 n개일 때 최소한 n개의 training example이 필요하다. 결론적으로, smoothness prior만 가정한 경우 모든 입력 공간을 k개의 서로 다른 영역들로 구분하기 위해선 k개의 example이 필요하다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61bce5ac-132e-401c-9c2a-3570d2c96455/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61bce5ac-132e-401c-9c2a-3570d2c96455/Untitled.png)

복잡한 함수를 효율적으로 표현하는 것과 추정된 함수가 새 입력들에 잘 일반화되는 것을 동시에 추구하는 것은 가능한 일이다. 이것이 달성되기 위해서는 data generating distribution에 대한 추가적인 가정들을 도입하고, 각 영역들에 의존성을 도입해야 한다. In this way, we can actually generalize non-locally.(local generalization이 smoothing인듯)

Deep learning 알고리즘 중에는 광범위한 ML 과제들에 적합한 암묵적, 명시적 가정을 도입해서 좋은 성과를 냈다.(대표적으로 weight을 공유하는 Convolution layer가 있겠죠?

DL의 core idea는 데이터가 다층 구조의 composition of factors/features로 이루어져있다고 가정하는 것이다. 

### 5.11.3 Manifold(다양체) Learning

다양체란 연결된 영역, 각 점 주변의 이웃과 연관된 점들의 집합을 뜻한다. 예를 들어 지구는 3차원 공간 안의 구면 다양체다. 

ML에서는 높은 차원의 공간에 내장된, 낮은 차원/자유도(degree of freedom)으로도 잘 근사할 수 있는 연결된 점들을 뜻한다. 

만약 데이터에 포함된 "interesting variation"이 $\mathbb{R}^n$에 넓게 퍼져 담겨있다면, 많은 ML 알고리즘들은 가망이 없다. 대신, 흥미로운 입력(유의미한 정보들?)이 일부 점들로 이루어진 몇몇 다양체에만 존재한다고 가정하여 이 어려움을 극복한다. 

상술한 개념을 Manifold hypothesis(다양체 가설)이라고 한다. 이미지, 텍스트 문자열, 음향 등에 관한 확률 분포도 각각 고유의 manifold 위에 있을 것이다. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/428476c4-6ae6-49ff-84b0-6dfc541a1a03/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/428476c4-6ae6-49ff-84b0-6dfc541a1a03/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cbfba0ec-e7b3-42c7-b861-dfa8309c074e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cbfba0ec-e7b3-42c7-b861-dfa8309c074e/Untitled.png)

random image vs face dataset

자료가   $\mathbb{R}^n$ 공간에 놓여 있다면, 기계학습 알고리즘이 그러한 자료를 그 자료의 특성에 잘 담는 다양체를 기준으로 하는 좌표로 표현하는 것이 더 좋다. 예를 들어, 3차원 공간의 도로를 1차원 도로번호로 지칭하는 것이 훨씬 좋다. 

책의 뒷부분에선 다양한 다양체 구조를 성공적으로 학습하는 기계학습 알고리즘이 소개될 것이다.
