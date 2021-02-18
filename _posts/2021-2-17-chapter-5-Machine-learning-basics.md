기계학습의 가장 중요한 기본 원리들을 소개한다. 


# 5.1 Learning Algorithms

학습 알고리즘이란 어떤 task들의 class $T$에 대한 알고리즘의 성능이 $P$라고 할 때, 경험 $E$를 이용하여 $P$를 향상시킬 수 있는 알고리즘을 말한다. 이 장에서는 $T$, $P$, $E$에 대하여 소개한 후 간단한 예시인 linear regression에 적용하는 과정을 소개한다.

## 5.1.1 The Task, $T$

학습 알고리즘을 이용하여 해결할 수 있는 task는 여러가지가 있다. 여기에서는 그 중 가장 자주 등장하는 task들의 예시를 소개한다.

- Classification
    - Input을 여러개의 class 중 하나로 분류하는 문제.
- Classification with missing input
    - Classification 이지만 input 중 일부가 없을 수 있는 문제.
    - Input 차원이 $n$이라고 할 때, input이 소실될 수 있는 모든 가능성을 고려한다면 $2^n$개의 함수를 알아야 한다. 하지만 학습 알고리즘을 사용하면, marginal distribution을 학습하여 하나의 함수만으로 추론할 수 있다.
    - 이러한 문제는 주로 medical domain같이 data의 값이 비싸고 얻기 어려운 분야에서 에서 풀어야 하는 경우가 많다.
- Regression
    - 분류의 continuous 버전. Input에 대하여 값을 예측하는 문제.
- Transcription
    - 사진 등의 구조적이지 않은 input에 대하여 텍스트를 뽑아내는 task들. 예를들어 사진에서 문장을 추출하는 task, 또는 음성인식이 transcription에 해당한다.
- Machine translation
    - 자연어 사이의 기계번역 등.
- Structured output
    - 알고리즘의 출력이 벡터이고 벡터의 성분들 사이에 중요한 관계가 있는 task. 다른 task들 중 여럿을 포함하는 넓은 범주이다.
    - 예를들어 input이 이미지이고 output이 이미지에 부여할 제목(자연어)인 경우도 이 task에 해당한다.
- Anomaly detection
    - 알고리즘이 데이터로부터 이상치를 찾아내는 task.
- Synthesis and sampling
    - Generative model들이 여기에 해당한다.
- Imputation of missing values
    - Input $\vec{x} \in R^n$ 중 빠져있는 일부 성분들을 예측하는 task.
- Denoising
    - Input에 포함된 noise를 제거하는 알고리즘.
    - 일반적으로 noise의 source를 모르기 때문에 데이터로부터 추론해야 한다.
- Density estimation
    - 데이터의 확률 밀도함수 또는 확률 질량함수를 추정하는 문제.
    - 이론적으로 데이터의 분포를 추정했다면 위에 소개한 다를 문제들은 이미 풀었다고 봐도 무방하다.

## 5.1.2  The Performance Measure, $P$

- $P$를 측정하는 궁극적인 목표는 알고리즘이 실제로 사용될 때의 성능을 가늠하기 위함이다. 이를 위해 학습에 사용하지 않은 test 데이터셋에 대하여 $P$를 측정한다.
- $P$를 측정하는 방법을 일반화하기는 어렵다. 결국 $P$를 통해 측정하려 하는 것은 알고리즘의 오류율 (또는 정확도) 이다. ***5.1.1***에서 소개한 것 처럼 많은 종류의 task들이 있고, 각 task에 적합한 $P$가 있다.

## 5.1.3 The Experience, $E$

- $E$를 두 종류로 분류하자면 지도학습과 비지도학습이 있다.
    - 지도학습의 목표는 주어진 input에 대한 예측을 정확히 하는 것이다. 즉, 알고리즘이 $p(y|x)$를 학습하는 것이 목적이다. 지도학습에 사용되는 데이터셋에는 label이 존재한다. Classification, regression등의 task가 지도학습을 통해 해결 가능하다.
    - 비지도학습의 목표는 주어진 자료 집합에 존재하는 구조적인 특성이나 분포를 학습하는 것이다. 이 경우 데이터셋에 label이 없다. 알고리즘은 주어진 input들만을 이용하여 $p(x)$ 자체를 학습하는 것이 목표이다. Clustering같은 task가 비지도학습으로 해결할 수 있다.
- $E$를 지도학습과 비지도학습으로 나누기는 하지만, 엄밀하게 정의된 개념들은 아니다. 지도학습 문제를 비지도학습으로 변환할 수도, 그 반대도 가능하다.
    - 비지도학습은 여러개의 지도학습 문제로 쪼개는 것이 가능하다.
    $p(\vec{x}) = \prod_i p(x_i|x_1, x_2, ..., x_{i-1})$
    - 지도학습 역시 비지도학습 알고리즘을 통해 전체 분포 $p(x, y)$를 학습한 후 조건부 확률을 계산하는 식으로 추론이 가능하다.
    $p(y|x) = \frac{p(x, y)}{\sum_{y'}p(x, y')}$
- 이 외에도 준지도학습, 다중 인스턴스 학습, 강화학습 등의 학습 방법들도 존재한다.
    - 준지도학습의 경우 데이터셋의 일부에만 label이 있다.
    - 다중 인스턴스 학습(multi-instance learning)의 경우 데이터셋에 특정 부류의 샘플이 포함되어 있다는 정보는 있지만 label은 없다.
    - 강화학습은 데이터셋이 고정되어 있지 않고 주변 환경과 지속적으로 상호작용한다. 이 책에서는 다루지 않는다.
- 학습 알고리즘을 다룰 때 데이터셋을 design matrix로 표현하는 경우가 많다. Design matrix는 데이터를 행렬 형태로 모아놓은 것인데, $i$번째 데이터 샘플을 design matrix $X$의 $i$번째 row로 표현한다.
$X = [\vec{x}_1, \vec{x}_2, ...]^T$

## 5.1.4 Example: Linear Regression

지금까지 학습 알고리즘의 관점에서 task, performance measure, experience가 무엇인지 소개하였다. 이번에는 이 개념들을 적용하여 선형 회귀 모델을 기술해보려 한다. 즉, task가 regression인 경우이다. 우선 선형 회귀 모델은 아래와 같이 생겼다.

$$\hat{y} = \vec{w}^T \vec{x}$$

선형 회귀 모델의 학습은 적절한 $\vec{w}$를 찾는 것이다.

이제 design matrix $X$를 이용하여 선형 회귀 모델의 적절한 $\vec{w}$를 찾아보자. 여러가지 방법이 있지만, performance measure는 평균제곱오차를 사용하려 한다. 즉, 이제 목표는 $X$에 대하여 다음의 값을 줄이는 것이다.

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m(\hat{y} - y) \\ = \frac{1}{m} ||\hat{y} - y||_2^2$$

선형 회귀 문제의 경우 MSE를 미분하여 closed form solution을 얻을 수 있다.

$$\nabla_{\vec{w}} MSE = 0 \\ \Rightarrow \vec{w} = (X^TX)^{-1} X^Ty$$

물론 ***5.1.2***에서 소개한 것 처럼, 이 모델이 실제로 사용될 때 성능을 가늠하기 위해 design matrix를 $X^{train}, X^{test}$로 분리한 후 $X^{train}$을 이용하여 찾은 $\vec{w}^{train}$의 성능을 $X^{test}$에 대하여 확인하는 것이 바람직하다.

# 5.2 Capacity, Overfitting and Underfitting

이 장에서는 기계학습 모델의 수용력, 과적합, 과소적합에 대하여 다룬다. 약간 추상적인 면이 있지만 이러한 개념들을 알아두는 것이 이후의 기계학습 모델을 공부할 때 도움이 될 것 같다.

- 과소적합, 과대적합
    - ***5.1.2***에서 말한 것 처럼 학습 알고리즘의 궁극적인 목표는 test set에 대한 오차를 줄이는 것 이다. 즉, 우리는 기계학습 모델을 학습시키며 두 가지를 신경써야 한다.
        1. Training set에 대한 오차를 작게 만드는 능력
        2. Test set에 대한 오차를 작게 만드는 능력
    - 1번은 만족했지만 2번은 만족하지 못한 경우를 과적합, 그 반대인 경우를 과소적합이라 부른다.

![_config.yml]({{ site.baseurl }}/assets/ch5/overfitting.png)

- 모델의 수용력
    - 모델의 가설 공간이라고도 부른다.
    - 만약 모델의 표현 수용력이 너무 크다면 과적합이 발생할 가능성이 높고, 표현 수용력이 너무 작다면 과소적합이 발생할 가능성이 높다.
    - 최적화에 사용한 알고리즘의 한계 등으로 모델이 실제로 표현할 수 있는 것이 표현 수용력보다 작을 수 있으며, 이를 모델의 유효 수용력이라 한다. 이러한 모델의 수용력을 수치화하는 다양한 방법이 있으며 그 중 하나는 VC dimension(Vapnik & Chervonenkis dimension)이다.
    - 모델의 수용력은 주로 모델의 매개변수에 의존한다. 예를들어 regression 을 할 때 몇차항까지 사용할지 등이 이에 해당된다.
    - 이러한 논의들을 딥러닝에 바로 적용하기는 어렵다. 딥러닝에는 여러가지 nonconvex함수와 최적화 알고리즘들이 관여하는데, 이러한 상황에서의 최적화 문제를 아직 잘 이해하지 못했기 때문이다.
- Bayes error
    - 만약 우리가 정말 우연히 자료 생성의 실제 확률분포를 표현하는 모델을 발견했다고 하자. 하지만 이 경우에도 데이터셋에 노이즈가 존재하기 때문에 오차는 0이 아니다.
    - 즉, 이렇게 데이터 자체에 내재되어있는 오차를 베이즈 오차라고 부른다.

## 5.2.1 The No Free Lunch Theorem

통계이론에 따르면, 아주 이상적인 상황에서 기계학습 알고리즘과 모든 데이터포인트를 하나의 값으로 분류하는 모델의 성능은 동일하다. 이 장에서는 그럼에도 불구하고 기계학습 알고리즘이 현실세계에서 잘 작동하는 근거를 제시한다.

- 귀납추론의 한계
    - 귀납추론의 관점에서 생각해보면 유한한 크기의 데이터셋을 이용하여 일반적인 규칙을 알아내는 것은 논리적으로 모순이다.
    - 즉, 현실세계에서 기계 학습 알고리즘을 이용하여 일반적인 규칙을 알아내는 것은 불가능하다.
    - 기계학습은 이러한 문제점을 거의 대부분의 데이터에 대하여 정확할 가능성이 있는 규칙을 찾음으로써 피해간다. 즉, 확률적으로 옳을 가능성이 있는 규칙을 추론하는 것이다.
- 공짜 점심 없음 정리
    - 공짜 점심 없음 정리란, 모든 가능한 데이터 분포에 대해 모델의 평균 성능을 계산할 경우 아직 관측하지 못한 데이터에 대한 모델들의 성능은 모두 동일하다.
    - 기계학습 알고리즘에 기대하는 것은 아직 관측하지 못한 test data에 대하여 좋은 성능을 얻는 것이다. 이러한 관점에서 모든 데이터포인트를 하나의 값으로 분류하는 모델과 굉장히 복잡한 기계학습 알고리즘의 성능은 동일하다.
    - 하지만 현실에서 모든 데이터 분포에 대하여 모델의 성능을 평가하는 것은 불가능하므로 우리가 관측한 분포 내에서 (구체적인 task에 대하여) 상대적으로 잘 작동하는 기계학습 알고리즘을 만드는 것은 가능하다. 즉, 우리의 목표는 현실세계에서 잘 작동하는 모델을 찾아내는 것이다.

## 5.2.2 Regularization

- ***5.2***의 초입에서 기계학습 알고리즘의 궁극적인 목표는 일반화 오차를 작게 하는 것이라고 소개하였다. 그리고 이러한 일반화 오차를 알고리즘의 표현력 관점에서 설명하였다.
- ***5.2***에서 설명한 모델의 가설공간이란 함수들의 집함 정도로 생각하면 된다. 우리가 기계학습 알고리즘을 학습한다는 것은 최적화 알고리즘을 통해 가설공간의 한 함수를 골라내는 것이다. 즉, 가설공간이 너무 크면 과적합 된 모델을 찾아낼 가능성이, 가설공간이 너무 작으면 과소적합된 모델을 찾아낼 가능성이 크다.
- 하지만, 우리가 해결하려는 task에 대한 사전정보를 알고있다면 모델의 특정한 가설공간에 선호도를 부여하여 과적합을 예방할 수 있다. 기계학습에서는 특정한 형태의 penelty를 부여하여 가설공간에 선호도를 부여하는데, 이를 정칙화(regularization)이라고 부른다.
    - 추후에 등장하겠지만 이러한 정칙화 항은 maximum a posteriori를 통해 모델을 선택할 때 prior distribution에 해당한다. 즉, 모델이 대충 어떻게 생겼을지 미리 알고있다는 의미이다.
- 기계학습 알고리즘에서 자주 쓰이는 정칙화 항들이 있는데 두가지만 소개하려 한다.
    - $L_1$-regulariztion: 모델을 학습할 때 loss에 모델 파라미터들의 $L_1$-norm을 더해준다. 학습 과정에서 모델은 선형적으로 파라미터들을 작은 값으로 선택하려 하고 sparse한 모델이 학습된다. Maximum a posteriori 관점에서 prior distribution을 Laplace distribution으로 선택한 것과 같다. 선형 회귀 모델에 $L_1$-regulariztion을 적용한 예시가 아래 그림에 있다. Regularization의 강도 $\lambda$를 조절함에 따라 모델이 과소적합, 과적합 되는 것을 방지할 수 있다.
    - $L_2$-regulariztion: 모델을 학습할 때 loss에 모델 파라미터들의 $L_2$-norm을 더해준다. $L_1$-norm과 동일하게 학습 과정에서 모델은 파라미터들을 작은 값으로 선택하려 하지만 상대적으로 큰 파라미터들에 대하여 더 큰 penelty를 부여한다. 결론적으로 모델은 적당히 작은 파라미터들을 선택하게 된다. Maximum a posteriori 관점에서 prior distribution을 Gaussian distribution으로 선택한 것과 같다.

![_config.yml]({{ site.baseurl }}/assets/ch5/regularization.png)


# 5.3 Hyperparameters and Validation Sets

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

# 5.4 Estimators, Bias and Variance

- 머신러닝에서 유용하게 사용되는 통계적인 도구들에 대한 소개

### Point Estimation

- 단 하나의 best 예측을 뽑기 위한 방법
- 대상은 하나의 파라미터일 수도, linear regression 같이 벡터 파라미터일수도 있음
- $\hat{\theta}$ : 파라미터 $\theta$의 estimator, true $\theta$에 가까울수록 좋은 estimator임

### Function Estimation

- 입력 벡터 $x$에 대한 $y$를 예측하는 경우, 해당 함수에 대한 estimation
- 함수 공간에서의 point estimation이라 할 수 있음

### Bias

- data들로부터 추정한 $\hat\theta_m$의 기댓값과 true $\theta$와의 차이
![_config.yml]({{ site.baseurl }}/assets/ch5/eq5_20.PNG)
- unbiased: $E(\hat\theta_m) = \theta$
- asymptotically unbiased: 데이터 갯수가 무한대로 가면 unbiased 해지는 경우

### Example: Bernoulli Distribution

- mean이 unbiased estimator가 됨

![_config.yml]({{ site.baseurl }}/assets/ch5/Bernoulli.PNG)

### Example: Gaussian Distribution Estimator of the Mean

- sample mean이 unbiased estimator가 됨

![_config.yml]({{ site.baseurl }}/assets/ch5/Gaussian.PNG)

### Example: Estimators of the Variance of a Gaussian Distribution

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

### Example: Bernoulli Distribution
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

# 5.5 Maximum Likelihood(ML) Estimation

실제 데이터의 분포 $p_{data}(\mathsf{\boldsymbol x})$로 부터 독립적으로 얻어진 $m$개의 샘플 $\mathbb{X} = \{\boldsymbol x^{(1)},...\ , \boldsymbol x^{(m)}\}$가 주어졌을 때, $p_{data}(\mathsf{\boldsymbol x})$를 추정하는 확률분포 $p_{model}(\mathsf{\boldsymbol x};\boldsymbol\theta)$를 생각하자. $p_{model}$을 maximize 시키는 $\boldsymbol\theta$는 다음과 같이 주어진다.

$$\boldsymbol\theta_{ML} = argmax_{\boldsymbol\theta} p_{model}(\mathbb{X};\boldsymbol\theta) = argmax_{\boldsymbol\theta} \prod_{i = 1}^m  p_{model}(\boldsymbol x^{(i)};\boldsymbol\theta)$$

위와 같은 곱셈은 underflow가 나기 쉬우므로 로그를 취하고,  $m$으로 나누어 기대값으로 바꿔준다.

$$\boldsymbol\theta_{ML} = argmax_{\boldsymbol\theta} \mathbb{E}_{\boldsymbol x \sim \hat p_{data}} \log p_{model}(\boldsymbol x;\boldsymbol\theta)$$

이러한 maximization과정은 $D_{KL} = \mathbb{E}_{\mathsf{\boldsymbol x}\sim\hat p_{data}}[\log \hat p_{data}(\boldsymbol x) - \log p_{model}(\boldsymbol x)]$를 $p_{model}$에 대해 minimize 하는 것과 동일하며, 즉, cross-entropy와도 연결된다.

## 5.5.1 Conditional Log-Likelihood and Mean Squared Error

$X$가 input, $Y$가 target인 일반적인 supervised learning을 생각하면 conditional maximum likelihood estimator는 다음과 같다.

$$\boldsymbol\theta_{ML} = argmax_{\boldsymbol \theta}P(\boldsymbol Y|\boldsymbol X ;\boldsymbol\theta)$$

모든 샘플이 i.i.d라면, 

$$\boldsymbol\theta_{ML} = argmax_{\boldsymbol\theta}\sum_{i=1}^m\log P(\boldsymbol y^{(i)}|\boldsymbol x^{(i)} ;\boldsymbol\theta)$$

- Linear Regression as Maximum Likelihood

    Input이 $\boldsymbol x$이고 output이 $\hat y$인 선형회귀에서 $p(y|\boldsymbol x) = \mathcal{N}(y;\hat y(\boldsymbol x;\boldsymbol w),\sigma^2)$라 하자.(즉, 선형회귀를 통해 Gaussian의 평균값을 예측)

    $$\sum_{i=1}^m\log p(y^{(i)}|\boldsymbol x^{(i)} ;\boldsymbol\theta) = -m\log\sigma -\frac{m}{2}\log(2\pi)-\sum_{i=1}^m\frac{\|\hat y^{(i)}-y^{(i)}\|^2}{2\sigma^2}$$

    위의 식을 보면 알수있듯 log-likelihood를 파라미터 $w$에 대해 maximize하는 과정은 결국 MSE loss를 minimize하는 과정과 일치한다.

## 5.5.2 Properties of Maximum Likelihood

ML estimator가 consistency를 가지려면 두가지 조건을 만족해야 한다.

- $p_{data}$ 가 model family $p_{model}(\cdot;\boldsymbol\theta)$에 속해야 한다.
- $p_{data}$에 대응되는 파라미터 $\boldsymbol\theta$가 여러개라면 어떤 $\boldsymbol\theta$가 데이터 생성 과정을 결정하는지 알 수 없으므로, $p_{data}$는 하나의 파라미터 $\boldsymbol\theta$에만 대응되야 한다.

# 5.6 Bayesian Statistics

Frequentist와는 다르게 파라미터 $\boldsymbol\theta$를 고정된 값이 아닌 확률변수로 보는 관점.

베이지안 통계에서는 $\boldsymbol\theta$에 대한 확률분포 $p(\boldsymbol\theta)$를 일종의 선입견(prior)으로 생각하고 어떤 사건이 일어날 확률을 계산한다. (prior는 불확실성이 높은 uniform distribution 또는 Gaussian distribution을 주로 이용한다고 한다.)

- $m$개의 데이터 샘플 $\{x^{(1)}, ...\ ,x^{(m)}\}$이 주어진 후의 $\boldsymbol{\theta}$에 대한 조건부 확률분포(posterior) $p(\boldsymbol{\theta}|x^{(1)}, ...\ ,x^{(m)})$은 Bayes rule로 부터 주어진다.

$$p(\boldsymbol\theta|x^{(1)}, ...\ ,x^{(m)}) = \frac{p(x^{(1)},...\ ,x^{(m)}|\boldsymbol\theta)p(\boldsymbol\theta)}{p(x^{(1)},...\ ,x^{(m)})}$$

- 확률이 가장 높은 하나의 $\theta$ 값을 구하는 ML과는 다르게 베이지안 통계에서는 가능한 모든 $\theta$값을 고려하여 확률을 계산한다. 예를들어, 새로운 데이터 $x^{(m+1)}$가 일어날 확률은 posterior를 weight로하여 아래와 같이 계산된다.
s
    $$p(x^{(m+1)}|x^{(1)}, ...\ ,x^{(m)}) = \int p(x^{(m+1)}|\boldsymbol\theta)p(\boldsymbol\theta|x^{(1)}, ...\ ,x^{(m)})d\boldsymbol\theta$$

## 5.6.1 Maximum A Posteriori(MAP) Estimation

연산이 간단한 ML estimation과 베이지안 통계의 prior를 짬뽕시킨 방법. 즉, prior가 point extimation에 영향을 미치도록함으로써 베이지안 통계의 이점을 얻음.

- 기존의 ML와는 다르게 posterior를 maximize 시킨다.

$$\boldsymbol\theta_{MAP} = argmax_{\boldsymbol\theta} p(\boldsymbol\theta|x) = argmax_{\boldsymbol\theta} \log p(x|\boldsymbol\theta)+\log p(\boldsymbol\theta)$$

- 우변의 첫번째 항은 ML과 동일하므로, MAP는 ML learning에 regularization 항($\log p(\boldsymbol\theta)$)을 추가한 것으로 해석 가능하다.

# 5.7 Supervised Learning Algorithm

## 5.7.1 Probabilistic Supervised Learning

데이터 $\boldsymbol x$ 와 그에 대한 정답 $y$가 주어졌을 때, 대부분의 지도학습은 확률분포 $p(y\mid \boldsymbol x)$를 추정하는 과정. 

- 선형회귀 문제에선  normal distribution으로 주어진다.

$$p(y\mid \boldsymbol x;\boldsymbol \theta) = \mathcal{N}(y;\boldsymbol \theta^{\top}\boldsymbol x,\boldsymbol I)$$

- $y$가 0 또는 1과 같이 이항 변수인 경우에는 시그모이드 함수를 사용하여 아래와 같이 표현 가능하고, 흔히 logistic regression이라고 불린다.

$$p(y=1\mid \boldsymbol x;\boldsymbol \theta) = \sigma(\boldsymbol\theta^{\top}\boldsymbol x)$$

## 5.7.2 Support Vector Machine

아래와 같이 주어진 데이터를 클래스로 나누어 분류하는 경우 선형 식 $\boldsymbol w^{\top}\boldsymbol x+b$ (빨간선) 을 기준으로 분류하는 방법.

![_config.yml]({{ site.baseurl }}/assets/ch5/svm_linear.png)

- 두 점선 위에 있는 세 점이 서포트 벡터이며, 두 점선 사이의 거리(gap 또는 margin)가 최대가 되도록 서포트 벡터를 정한다.
- 문제는 주어진 데이터가 위의 예시처럼 선형으로 분류가 안되는 경우인데, 이때는 커널이라는 것을 사용하여 다시 선형 분류 문제로 바꾸어 해결 가능.

    $$\boldsymbol w^{\top}\boldsymbol x + b=b\ +\sum_{i=1}^m \alpha_i\boldsymbol x^{\top}\boldsymbol x^{(i)}$$

    위의 식에서 $\boldsymbol x$ 를 feature function $\phi(\boldsymbol x)$로 변환하고, 커널 $k(\boldsymbol x, \boldsymbol x^{(i)}) = \phi(\boldsymbol x) \cdot \phi(\boldsymbol x^{(i)})$를 정의하면 아래와 같이 식 변형 가능.

    $$f(\boldsymbol x) = b + \sum_{i=1}^m \alpha_i k(\boldsymbol x, \boldsymbol x^{(i)})$$

    즉, $f(\boldsymbol x)$라는 초평면으로 클래스 분류가 가능해진다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/svm_kernel.png)

## 5.7.3 Other Simple Supervised Learning Algorithms

- k-nearest neighbors

    새로운 데이터가 주어졌을 때, 가장 가까운 k개의 이웃 데이터들의 class 중 비율이 높은 class를 선택.

![_config.yml]({{ site.baseurl }}/assets/ch5/knn.png)

- decision tree

    주어진 데이터에 대해 연속적인 이진분류를 하여 class를 나누는 방법.

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-4.png)

# 5.8 Unsupervised Learning Algorithms

비지도 학습의 기본적인 목적은 데이터의 최상의 또는 심플한 표현법을 찾는 것. 일반적으로 아래의 세가지 방법을 사용한다.

- lower dimensional representations
- sparse representations
- independent representations

## 5.8.1 Principal Components Analysis

데이터를 표현하기 위한 디멘션을 줄이는 좋은 방법이며 elements들 사이의 선형 의존관계 또한 제거할 수 있다.

- m by n 행렬 $\boldsymbol X$ 의 principal component는 $\boldsymbol X^{\top}\boldsymbol X$ 의 eigenvectors로 주어지는데, SVD를 이용하여 $\boldsymbol X$를 표현하면  공분산은 아래와 같이 적히고

    $$\text{Var}[\boldsymbol x] = \frac{1}{m-1}\boldsymbol X^{\top}\boldsymbol X = \frac{1}{m-1}\boldsymbol {(U\Sigma W^{\top})}^{\top}\boldsymbol{U\Sigma W^{\top}} = \frac{1}{m-1}\boldsymbol W\boldsymbol\Sigma^2\boldsymbol W^{\top}$$

    데이터를 선형 변환($\boldsymbol z = \boldsymbol x^{\top}\boldsymbol W$) 하게되면 $\boldsymbol z$는 대각화 된 공분산을 가진다.

    $$\text{Var}[\boldsymbol z] = \frac{1}{m-1}\boldsymbol\Sigma^2$$

![_config.yml]({{ site.baseurl }}/assets/ch5/pca.png)

## 5.8.2 k-means Clustering

Sparse representation 중의 하나이며 주어진 데이터를 k개의 cluster로 나누는 방법.

- 즉, $\boldsymbol x$를 k-dimensional one-hot code로 표현할 수 있고, 아래의 알고리즘을 통해 k개의 centroids $\{\boldsymbol \mu^{(1)},...\ ,\boldsymbol\mu^{(k)}\}$를 결정한다.
    - 각각의 데이터들을 가장 가까운 centroid에 해당하는 cluster에 할당.
    - 각각의 centroid는 자신의 cluster에 속하는 데이터의 평균값으로 업데이트.
- 문제는 clustering이 잘 되었는지를 평가할 수 없다는 것.

# 5.9 Stochasitic Gradient Descent

좋은 일반화를 위해선 dataset의 크기가 커야하지만 동시에 dataset의 규모가 증가하면 계산비용이 커진다. 기계학습 알고리즘이 사용하는 cost function을 training examples of some per-example loss function으로 분해할 수 있다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-7.png)

여기서 $L$은 per-example loss $L(x,y,\theta) = -log\ p(y|x;\theta)$. 

Gradient descent를 이용하기 위해선 알고리즘의 cost function $\nabla_{\theta} J(\theta)$를 계산해야 한다. 연산의 코스트는 데이터의 수에 대해 선형의 시간복잡도를 가진다. 

SGD의 아이디어는 gradient가 하나의 기댓값이라는 점에서 출발한다. SGD에서는 training set에서 작은 데이터 단위 minibatch $\mathbb{B}$를 뽑아 학습에 사용한다. minibatch의 사이즈는 $m'$이며, 이 minibatch로 gradient를 추정하면 아래와 같다. 

$$g = {1 \over m'} \nabla_{\theta}\sum_{i=1}^{m'}L(x^{(i)} y^{(i)}, \theta)$$

예전엔 non-convex함수 최적화에 GD를 사용하지 않아야한다는 인식이 많았으나, 지금은 다름. 

m의 크기만 고정된다면 training data의 크기가 아무리 커져도 학습 비용이 $O(1)$이 될 것...?

# 5.10 Building a Machine Learning Algorithm

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

# 5.11 Challenges Motivating Deep Learning

Linear regression 등 5장에서 설명한 알고리즘들은 speech recognition, object recognition 등의 문제들은 잘 풀지 못한다. 이번 자에서는 복잡한 함수들을 배우는데 swallow model들이 적합하지 않다는 점 등을 다룬다. 

## 5.11.1 The curse of dimensionality(차원의 저주)

기계학습에서 자료의 차원이 아주 높을 때 풀기 어려워질 때가 많다. 변수의 개수에 따라 서로 다른 구성/조합의 개수가 지수함수적으로 증가한다. 차원의 저주가 초래하는 어려움 중 하나는 statistical challenge이다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled.png)

그림 설명: 1차원에서는 변수가 하나 뿐이고, 구별해야 할 변수의 다른 값들은 10개뿐이다. 그런데 2, 3차원으로 가면 지수적으로 구별해야할 값들이 증가한다. 1차원에서와 비슷하게 공간의 구멍들에 데이터를 채우려면 데이터의 양도 지수적으로 증가해야 한다. 

위 그림에서 보듯, x의 가능한 구성들의 개수가 training example의 수보다 훨씬 크면 statistical challenge가 발생한다. 데이터량이 한정적인 상황에서 차원이 높아지면 data가 하나도 배치되지 못한 격자 칸도 많아질 것이다. 

이렇게 차원의 저주에 걸린 경우 input data에 대해 어떻게 추론해야 할까? 전통적인 기계학습 알고리즘들은 new input data와 가장 가까이에 있는 training data의 출력와 이 new intput의 출력이 같을 것이라 가정한다. 

## 5.11.2 Local Constancy and Smoothness Regularization

ML 알고리즘의 잘 일반회되기 위해 알고리즘이 배워야 할 함수의 종류에 대한 prior belief를 알고리즘에 제공해야 한다. 많이 쓰이는 암묵적 prior로는 smoothness prior(local constancy prior)이 있다. 이 prior은 "함수가 작은 영역 안에서 아주 크게 변해서는 안 된다"는 제약을 뜻한다. 간단한 구조의 알고리즘들은 좋은 일반화를 보장하기 위한 수단이 이 prior밖에 없다. 

- smoothness prior이 충분한 prior이 될 수 없는 이유

smoothness prior은 특정 조건을 만족하는 함수 $f^*$가 아래와 같은 성질을 띄는 것을 뜻한다.

$$f^*(x) \approx f^*(x+\epsilon)$$

이런 smoothness prior의 극단적인 예로 k-nearest neighbor가 있다. k = 1일 때, the number of distinguishable regions은 be more than the number of training examples을 넘을 수 없다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-2.png)

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-3.png)

Decision tree 방법에도 smoothness만 의존하는 학습의 한계들이 적용된다. leaf node가 n개일 때 최소한 n개의 training example이 필요하다. 결론적으로, smoothness prior만 가정한 경우 모든 입력 공간을 k개의 서로 다른 영역들로 구분하기 위해선 k개의 example이 필요하다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-4.png)

복잡한 함수를 효율적으로 표현하는 것과 추정된 함수가 새 입력들에 잘 일반화되는 것을 동시에 추구하는 것은 가능한 일이다. 이것이 달성되기 위해서는 data generating distribution에 대한 추가적인 가정들을 도입하고, 각 영역들에 의존성을 도입해야 한다. In this way, we can actually generalize non-locally.(local generalization이 smoothing인듯)

Deep learning 알고리즘 중에는 광범위한 ML 과제들에 적합한 암묵적, 명시적 가정을 도입해서 좋은 성과를 냈다.(대표적으로 weight을 공유하는 Convolution layer가 있겠죠?

DL의 core idea는 데이터가 다층 구조의 composition of factors/features로 이루어져있다고 가정하는 것이다. 

## 5.11.3 Manifold(다양체) Learning

다양체란 연결된 영역, 각 점 주변의 이웃과 연관된 점들의 집합을 뜻한다. 예를 들어 지구는 3차원 공간 안의 구면 다양체다. 

ML에서는 높은 차원의 공간에 내장된, 낮은 차원/자유도(degree of freedom)으로도 잘 근사할 수 있는 연결된 점들을 뜻한다. 

만약 데이터에 포함된 "interesting variation"이 $\mathbb{R}^n$에 넓게 퍼져 담겨있다면, 많은 ML 알고리즘들은 가망이 없다. 대신, 흥미로운 입력(유의미한 정보들?)이 일부 점들로 이루어진 몇몇 다양체에만 존재한다고 가정하여 이 어려움을 극복한다. 

상술한 개념을 Manifold hypothesis(다양체 가설)이라고 한다. 이미지, 텍스트 문자열, 음향 등에 관한 확률 분포도 각각 고유의 manifold 위에 있을 것이다. 

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-5.png)

![_config.yml]({{ site.baseurl }}/assets/ch5/Untitled-6.png)

random image vs face dataset

자료가 $\mathbb{R}^n$ 공간에 놓여 있다면, 기계학습 알고리즘이 그러한 자료를 그 자료의 특성에 잘 담는 다양체를 기준으로 하는 좌표로 표현하는 것이 더 좋다. 예를 들어, 3차원 공간의 도로를 1차원 도로번호로 지칭하는 것이 훨씬 좋다. 

책의 뒷부분에선 다양한 다양체 구조를 성공적으로 학습하는 기계학습 알고리즘이 소개될 것이다.
