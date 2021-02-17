## 5.3 Hyperparameters and Validation Sets

- Hyperparameter: 모델이 학습하는게 아니라 사용자가 값을 설정하는 parameter로, 대부분의 머신러닝 알고리즘이가지고 있음
  - 예) polynomial regression에서 입력 feature의 차수, weight decay에서 <lambda>
- 최근 hyperparameter까지 최적화해주는 모델들이 나오기는 하지만, 일반적으로 적절한 값을 선택하는 것은 어려운 문제임
- 보통 여러 후보 값으로 조절해가며 모델의 성능을 테스트함 -> validation set 이용
  - validation set: test set이 아닌 training set의 일부로, 학습할 때 사용하지 않음
  - 일반적인 데이터 분배 비율: training : validation : test = 6 : 2 : 2

## 5.3.1 Cross-Validation

- 데이터를 위의 세 그룹으로 나누어 알고리즘을 평가할 때, test set의 크기가 충분히 크지 않은 경우엔 통계적 불확실성으로 인해 제대로 평가하기 부족함
- 따라서 데이터를 임의의 여러 그룹으로 나누어 평가를 반복하는 방법임
- k-fold cross-validation: 데이터를 서로 겹치지 않게 k개의 그룹으로 나누고, 각각의 그룹을 test set으로 사용하는 총 k번의 평가를 함

<fig 3개>

## 5.3.supp Monte-Carlo Cross Validation

- 일정 비율로 test set을 임의로 뽑아 테스트 하는 과정을 반복함
- 같은 데이터가 여러번 test set에 등장할 수 있다는 점이 k-fold cross validation과의 가장 큰 차이임

## 5.4 Estimators, Bias and Variance

- 머신러닝에서 유용하게 사용되는 통계적인 도구들에 대한 소개

## Point Estimation
- 단 하나의 best 예측을 뽑기 위한 방법
- 대상은 하나의 파라미터일 수도, linear regression 같이 벡터 파라미터일수도 있음
- \hat{theta} : 파라미터 \theta의 estimator, true \theta에 가까울수록 좋은 estimator임

## Function Estimation
- 입력 벡터 \x에 대한 \y를 예측하는 경우, 해당 함수에 대한 estimation
- 함수 공간에서의 point estimation이라 할 수 있음

## Bias
- data들로부터 추정한 ^θm의 기댓값과 true θ와의 차이
<Eq 5.20>
- unbiased: E(^θm) = theta
- asymptotically unbiased: 데이터 갯수가 무한대로 가면 unbiased 해지는 경우

## Example: Bernoulli Distribution
- mean이 unbiased estimator가 됨

## Example: Gaussian Distribution Estimator of the Mean
- sample mean이 unbiased estimator가 됨

## Example: Estimators of the Variance of a Gaussian Distribution
- sample variance는 true variance의 biased estimator임
- unbiased estimator를 만들기 위해서는 분모에 m대신 m-1을 사용해야 함

## 5.4.3 Variance and Standard Error
- Variance: 추산한 값이 데이터 샘플마다 얼마나 크게 달라지는지에 대한 지표, Var(^θ)
- Standard error: variance의 루트 값, SE(^θ)
- 낮은 bias, 낮은 variance를 가지는 estimator가 대체로 우리가 원하는 형태임
- 유한한 데이터로 통계적인 수치를 계산할 때는 항상 불확실성을 내포함
- 같은 distribution에서 얻어진 데이터들이라 하더라도, 통계치는 달라질 수 있고, 이 달라지는 정도를 정량화하기 위한 도구임
- 평균의 standard error:
<SE>
- 평균 u의 95% 신뢰 구간:
<confidence>
  - 알고리즘 A의 error에 대한 95퍼센트 신뢰구간의 upper bound가 알고리즘 B의 error에 대한 95퍼센트 신뢰구간의 lower bound보다 작다면 알고리즘 A가 더 낫다고 하곤 함

## Example: Bernoulli Distribution
- Bernoulli distribution의 estimator로 평균 값을 가정할 때, 데이터 수 m이 증가할수록 estimator의 variance는 감소함
<variance_bernoulli>

## 5.4.4 Trading oﬀ Bias and Variance to Minimize Mean Squared Error
- Bias와 variance는 estimator error의 서로 다른 원인임
  - Bias: 함수나 파라미터의 참 값에서 예상되는 편차를 측정함
  - Variance: 데이터 샘플링으로 인해 발생할 가능성이 있는 예측치 값과의 편차를 나타냄
  - 일반적으로 bias와 variance는 아래 그림과 같이 trade-off 관계를 가짐
  - 이 때 cross-validation을 이용해 bias와 variance의 변화를 살피는 방법이 널리 사용됨
<optimal capacity>
  - 또는 bias와 variance를 모두 포함하는 mean squared error (MSE)를 최소화하는 조건을 탐색함
<MSE>
  - 일반적으로 capacity가 증가하면 bias는 줄어들고 variance는 증가함

## 5.4.5 Consistency
- Training set의 크기가 증가하면 point estimates가 참값에 수렴하는 성질
<consistency>
- Consistency가 성립한다면 데이터 수가 증가할 때 bias가 줄어듬을 보장함
- 하지만 역은 항상 성립하지는 않음 -> bias가 줄어든다고 consistency가 성립하지는 않음
  - 예) Dataset에서 normal distribution의 평균을 추정할 때, 무조건 첫번째 샘플을 estimator로 사용한다면 unbiased이긴 하지만, 데이터 개수가 무한대가 될 때 unbiased 해지는 것은 아니므로 consistency라 할 수 없다.

## 5.5 Maximum Likelihood(ML) Estimation

실제 데이터의 분포 $p_{data}(\mathsf{\boldsymbol x})$로 부터 독립적으로 얻어진 $m$개의 샘플 $\mathbb{X} = \{\boldsymbol x^{(1)},...\ , \boldsymbol x^{(m)}\}$가 주어졌을 때, $p_{data}(\mathsf{\boldsymbol x})$를 추정하는 확률분포 $p_{model}(\mathsf{\boldsymbol x};\boldsymbol\theta)$를 생각하자. $p_{model}$을 maximize 시키는 $\boldsymbol\theta$는 다음과 같이 주어진다.

$$\boldsymbol\theta_{ML} = \argmax_{\boldsymbol\theta} p_{model}(\mathbb{X};\boldsymbol\theta) = \argmax_{\boldsymbol\theta} \prod_{i = 1}^m  p_{model}(\boldsymbol x^{(i)};\boldsymbol\theta)$$

위와 같은 곱셈은 underflow가 나기 쉬우므로 로그를 취하고,  $m$으로 나누어 기대값으로 바꿔준다.

$$\boldsymbol\theta_{ML} = \argmax_{\boldsymbol\theta} \mathbb{E}_{\boldsymbol x \sim \hat p_{data}} \log p_{model}(\boldsymbol x;\boldsymbol\theta)$$

이러한 maximization과정은 $D_{KL} = \mathbb{E}_{\mathsf{\boldsymbol x}\sim\hat p_{data}}[\log \hat p_{data}(\boldsymbol x) - \log p_{model}(\boldsymbol x)]$를 $p_{model}$에 대해 minimize 하는 것과 동일하며, 즉, cross-entropy와도 연결된다.

### 5.5.1 Conditional Log-Likelihood and Mean Squared Error

$X$가 input, $Y$가 target인 일반적인 supervised learning을 생각하면 conditional maximum likelihood estimator는 다음과 같다.

$$\boldsymbol\theta_{ML} = \argmax_{\boldsymbol \theta}P(\boldsymbol Y|\boldsymbol X ;\boldsymbol\theta)$$

모든 샘플이 i.i.d라면, 

$$\boldsymbol\theta_{ML} = \argmax_{\boldsymbol\theta}\sum_{i=1}^m\log P(\boldsymbol y^{(i)}|\boldsymbol x^{(i)} ;\boldsymbol\theta)$$

- Linear Regression as Maximum Likelihood

    Input이 $\boldsymbol x$이고 output이 $\hat y$인 선형회귀에서 $p(y|\boldsymbol x) = \mathcal{N}(y;\hat y(\boldsymbol x;\boldsymbol w),\sigma^2)$라 하자.(즉, 선형회귀를 통해 Gaussian의 평균값을 예측)

    $$\sum_{i=1}^m\log p(y^{(i)}|\boldsymbol x^{(i)} ;\boldsymbol\theta) = -m\log\sigma -\frac{m}{2}\log(2\pi)-\sum_{i=1}^m\frac{\|\hat y^{(i)}-y^{(i)}\|^2}{2\sigma^2}$$

    위의 식을 보면 알수있듯 log-likelihood를 파라미터 $w$에 대해 maximize하는 과정은 결국 MSE loss를 minimize하는 과정과 일치한다.

### 5.5.2 Properties of Maximum Likelihood

ML estimator가 consistency를 가지려면 두가지 조건을 만족해야 한다.

- $p_{data}$ 가 model family $p_{model}(\cdot;\boldsymbol\theta)$에 속해야 한다.
- $p_{data}$에 대응되는 파라미터 $\boldsymbol\theta$가 여러개라면 어떤 $\boldsymbol\theta$가 데이터 생성 과정을 결정하는지 알 수 없으므로, $p_{data}$는 하나의 파라미터 $\boldsymbol\theta$에만 대응되야 한다.

## 5.6 Bayesian Statistics

Frequentist와는 다르게 파라미터 $\boldsymbol\theta$를 고정된 값이 아닌 확률변수로 보는 관점.

베이지안 통계에서는 $\boldsymbol\theta$에 대한 확률분포 $p(\boldsymbol\theta)$를 일종의 선입견(prior)으로 생각하고 어떤 사건이 일어날 확률을 계산한다. (prior는 불확실성이 높은 uniform distribution 또는 Gaussian distribution을 주로 이용한다고 한다.)

- $m$개의 데이터 샘플 $\{x^{(1)}, ...\ ,x^{(m)}\}$이 주어진 후의 $\boldsymbol\theta$에 대한 조건부 확률분포(posterior) $p(\boldsymbol\theta|x^{(1)},...\ ,x^{(m)})$은 Bayes rule로 부터 주어진다.

$$p(\boldsymbol\theta|x^{(1)}, ...\ ,x^{(m)}) = \frac{p(x^{(1)},...\ ,x^{(m)}|\boldsymbol\theta)p(\boldsymbol\theta)}{p(x^{(1)},...\ ,x^{(m)})}$$

- 확률이 가장 높은 하나의 $\theta$ 값을 구하는 ML과는 다르게 베이지안 통계에서는 가능한 모든 $\theta$값을 고려하여 확률을 계산한다. 예를들어, 새로운 데이터 $x^{(m+1)}$가 일어날 확률은 posterior를 weight로하여 아래와 같이 계산된다.

    $$p(x^{(m+1)}|x^{(1)}, ...\ ,x^{(m)}) = \int p(x^{(m+1)}|\boldsymbol\theta)p(\boldsymbol\theta|x^{(1)}, ...\ ,x^{(m)})d\boldsymbol\theta$$

### 5.6.1 Maximum A Posteriori(MAP) Estimation

연산이 간단한 ML estimation과 베이지안 통계의 prior를 짬뽕시킨 방법. 즉, prior가 point extimation에 영향을 미치도록함으로써 베이지안 통계의 이점을 얻음.

- 기존의 ML와는 다르게 posterior를 maximize 시킨다.

$$\boldsymbol\theta_{MAP} = \argmax_{\boldsymbol\theta} p(\boldsymbol\theta|x) = \argmax_{\boldsymbol\theta} \log p(x|\boldsymbol\theta)+\log p(\boldsymbol\theta)$$

- 우변의 첫번째 항은 ML과 동일하므로, MAP는 ML learning에 regularization 항($\log p(\boldsymbol\theta)$)을 추가한 것으로 해석 가능하다.

## 5.7 Supervised Learning Algorithm

### 5.7.1 Probabilistic Supervised Learning

데이터 $\boldsymbol x$ 와 그에 대한 정답 $y$가 주어졌을 때, 대부분의 지도학습은 확률분포 $p(y\mid \boldsymbol x)$를 추정하는 과정. 

- 선형회귀 문제에선  normal distribution으로 주어진다.

$$p(y\mid \boldsymbol x;\boldsymbol \theta) = \mathcal{N}(y;\boldsymbol \theta^{\top}\boldsymbol x,\boldsymbol I)$$

- $y$가 0 또는 1과 같이 이항 변수인 경우에는 시그모이드 함수를 사용하여 아래와 같이 표현 가능하고, 흔히 logistic regression이라고 불린다.

$$p(y=1\mid \boldsymbol x;\boldsymbol \theta) = \sigma(\boldsymbol\theta^{\top}\boldsymbol x)$$

### 5.7.2 Support Vector Machine

아래와 같이 주어진 데이터를 클래스로 나누어 분류하는 경우 선형 식 $\boldsymbol w^{\top}\boldsymbol x+b$ (빨간선) 을 기준으로 분류하는 방법.

![_config.yml]({{ site.baseurl }}/assets/ch5/svm_linear.png)

- 두 점선 위에 있는 세 점이 서포트 벡터이며, 두 점선 사이의 거리(gap 또는 margin)가 최대가 되도록 서포트 벡터를 정한다.
- 문제는 주어진 데이터가 위의 예시처럼 선형으로 분류가 안되는 경우인데, 이때는 커널이라는 것을 사용하여 다시 선형 분류 문제로 바꾸어 해결 가능.

    $$\boldsymbol w^{\top}\boldsymbol x + b=b\ +\sum_{i=1}^m \alpha_i\boldsymbol x^{\top}\boldsymbol x^{(i)}$$

    위의 식에서 $\boldsymbol x$ 를 feature function $\phi(\boldsymbol x)$로 변환하고, 커널 $k(\boldsymbol x, \boldsymbol x^{(i)}) = \phi(\boldsymbol x) \cdot \phi(\boldsymbol x^{(i)})$를 정의하면 아래와 같이 식 변형 가능.

    $$f(\boldsymbol x) = b + \sum_{i=1}^m \alpha_i k(\boldsymbol x, \boldsymbol x^{(i)})$$

    즉, $f(\boldsymbol x)$라는 초평면으로 클래스 분류가 가능해진다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/svm_kernel.png)

### 5.7.3 Other Simple Supervised Learning Algorithms

- k-nearest neighbors

    새로운 데이터가 주어졌을 때, 가장 가까운 k개의 이웃 데이터들의 class 중 비율이 높은 class를 선택.

![_config.yml]({{ site.baseurl }}/assets/ch5/knn.png)

- decision tree

    주어진 데이터에 대해 연속적인 이진분류를 하여 class를 나누는 방법.

![_config.yml]({{ site.baseurl }}/assets/ch5/decision_tree.png)

## 5.8 Unsupervised Learning Algorithms

비지도 학습의 기본적인 목적은 데이터의 최상의 또는 심플한 표현법을 찾는 것. 일반적으로 아래의 세가지 방법을 사용한다.

- lower dimensional representations
- sparse representations
- independent representations

### 5.8.1 Principal Components Analysis

데이터를 표현하기 위한 디멘션을 줄이는 좋은 방법이며 elements들 사이의 선형 의존관계 또한 제거할 수 있다.

- m by n 행렬 $\boldsymbol X$ 의 principal component는 $\boldsymbol X^{\top}\boldsymbol X$ 의 eigenvectors로 주어지는데, SVD를 이용하여 $\boldsymbol X$를 표현하면  공분산은 아래와 같이 적히고

    $$\text{Var}[\boldsymbol x] = \frac{1}{m-1}\boldsymbol X^{\top}\boldsymbol X = \frac{1}{m-1}\boldsymbol {(U\Sigma W^{\top})}^{\top}\boldsymbol{U\Sigma W^{\top}} = \frac{1}{m-1}\boldsymbol W\boldsymbol\Sigma^2\boldsymbol W^{\top}$$

    데이터를 선형 변환($\boldsymbol z = \boldsymbol x^{\top}\boldsymbol W$) 하게되면 $\boldsymbol z$는 대각화 된 공분산을 가진다.

    $$\text{Var}[\boldsymbol z] = \frac{1}{m-1}\boldsymbol\Sigma^2$$

![_config.yml]({{ site.baseurl }}/assets/ch5/pca.png)

### 5.8.2 k-means Clustering

Sparse representation 중의 하나이며 주어진 데이터를 k개의 cluster로 나누는 방법.

- 즉, $\boldsymbol x$를 k-dimensional one-hot code로 표현할 수 있고, 아래의 알고리즘을 통해 k개의 centroids $\{\boldsymbol \mu^{(1)},...\ ,\boldsymbol\mu^{(k)}\}$를 결정한다.
    - 각각의 데이터들을 가장 가까운 centroid에 해당하는 cluster에 할당.
    - 각각의 centroid는 자신의 cluster에 속하는 데이터의 평균값으로 업데이트.
- 문제는 clustering이 잘 되었는지를 평가할 수 없다는 것.
