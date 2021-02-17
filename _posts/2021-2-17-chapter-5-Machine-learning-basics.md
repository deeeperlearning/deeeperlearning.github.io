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
