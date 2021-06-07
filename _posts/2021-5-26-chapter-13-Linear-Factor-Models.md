이번 단원에서는 잠재 변수를 가지는 간단한 확률론적 모델(probabilistic model)인 선형 인자 모델(linear factor model)에 대해 다룬다. 

선형 인자 모델을 통해 데이터를 생산하는 과정은 선형 디코더의 형태를 가진다.

$$\boldsymbol{\mathbf{h}} \sim p(\boldsymbol h)\ , \ \ \boldsymbol x = \boldsymbol W\boldsymbol h +\boldsymbol b+ \text{noise}  $$

노이즈는 일반적으로 대각 공분산을 가지는 가우시안으로 선택한다.

## 13.1 Probabilistic PCA and Factor Analysis

확률론적 PCA와 인자분석(factor analysis)은 선형 인자 모델과 동일한 수식을 따르지만 노이즈와 잠재 변수 $\boldsymbol h$의 prior 분포가 다를 뿐이다.

- 인자분석

    잠재 변수는 단위 분산을 가지는 가우시안을 따르며, 

    $$\boldsymbol{\mathbf{h}} \sim \mathcal N(\boldsymbol h;\boldsymbol0,\boldsymbol I) $$

    노이즈는 대각 공분산 $\boldsymbol \psi$ 을 가지는 가우시안을 따른다.

    $$\boldsymbol \psi =\text{diag}(\boldsymbol \sigma^2) \ \ \text{with}\ \ \boldsymbol \sigma^2 = [\sigma_1^2,\sigma_2^2,...,\sigma_n^2]^{\top}$$

    따라서 잠재 변수는 관찰 된 서로 다른 변수 $x_i$ 사이의 종속성을 찾는 역할을 하게되고, $\boldsymbol x$는 아래의 정규분포를 따르게 된다.

    $$\boldsymbol{\mathbf x} \sim \mathcal N(\boldsymbol x;\boldsymbol b, \boldsymbol{WW^{\top}}+\boldsymbol \psi )$$

- 확률론적 PCA

    잠재 변수는 인자분석 모델과 동일하지만 노이즈의 공분산만 다르다. ($\sigma_i^2 = \sigma^2$)

    $$\boldsymbol{\mathbf x} \sim \mathcal N(\boldsymbol x;\boldsymbol b, \boldsymbol{WW^{\top}}+\sigma^2\boldsymbol I ) $$

    분산 $\sigma^2$가 0이 되면 $\boldsymbol h$는 $\boldsymbol x - \boldsymbol b$를 $\boldsymbol W$로 projection 시킨 꼴이 되기 때문에 일반적인 PCA와 동일하다. 

## 13.2 Independent Component Analysis (ICA)

앞서 소개한 모델들과는 다르게, 독립 성분 분석은 $\boldsymbol h$의 각 성분이 서로 독립이 되도록 하는 방법이다.

방에서 두 사람이 말하고있고, 두개의 마이크를 통해 녹음이 되는 상황을 생각해보자. 두 마이크를 통해 녹음 된 신호를 각각 $x_1, \ x_2$라 하고, 두 사람의 음성을 각각 $h_1, \ h_2$라 하면 아래와 같이 적을 수 있다.

$$x_1 = w_{11}h_1 +w_{12}h_2 \\ x_2 = w_{21}h_1 +w_{22}h_2$$

즉, $\boldsymbol x = \boldsymbol{Wh}$로 적을 수 있고, 두 사람의 음성 벡터 $\boldsymbol h = \boldsymbol W^{-1}\boldsymbol x$를 찾아내는 것이 ICA의 목적이다.

- ICA는 다른 선형 요소 모델과는 다르게 $\boldsymbol h$의 각 성분이 non-Gaussian 분포를 따르도록 정한다. 만약  각 성분이 독립적인 가우시안 분포를 따른다면, 동일한 분포 $p(\boldsymbol x)$에 대해 여러개의 정답 $\boldsymbol W$가 존재할 수 있기 때문이다.
- 많은 ICA 방법들은 $\boldsymbol h$의 각 성분이 첨도가 높은 분포를 따르도록 하기 때문에 generative 모델보다는 신호들을 분리하는 도구로 많이 쓰인다.
- PCA를 비선형 오토인코더로 일반화 시키는 것과 같이, ICA를 비선형 generative 모델로 확장하는 것도 가능하다고 한다.
- ICA는 또한 통계적 의존성이 그룹 내에서는 허용되지만 그룹 간에는 권장되지 않도록 feature 그룹을 학습하는 모델로도 확장 가능하다.

## 13.3 Slow Feature Analysis

Slow feature analysis(SFA)는 invariant feature들을 학습하는 linear factor model이 slowness principle에 기반한다. 아이디어는 다음과 같다. scene들의 중요한 특성은 장면장면의 개별 특성보다 느리게 변한다는 것이다. 예를 들어 얼룩말이 이미지를 왼쪽에서 오른쪽으로 가로지를 때 얼룩말의 줄무늬는 각 픽셀에서 색과 검은색이 빠르게 교차하지만 얼룩말이 존재하는 지의 여부를 나타내는 특징은 변하지 않는다. 이런 시간에 따라 느리게 변하는 특징들을 학습하도록 모형을 regularize하는 것이 중요하다.

Slow feature을 model에 적용하는 방법은, 비용함수에 아래와 같은 항을 추가하는 것이다.

![_config.yml]({{ site.baseurl }}/assets/ch13/1.png)

$\lambda$는 slow feature regularization의 영향력을 결정하는 hyperparameter, t는 시간 순의 색인, $L$은 두 input간의 거리를 측정하는 손실함수, f는 regularize를 수행하는 feature extractor이다.

SFA 알고리즘은 $f(x;\theta)$를 하나의 선형 변환으로 정의하고 아래와 같은 최적화 문제를 푼다.

![_config.yml]({{ site.baseurl }}/assets/ch13/2.png)

여러개의 느린 특징을 학습하기 위해서는 다름 제약이 필요하다.

![_config.yml]({{ site.baseurl }}/assets/ch13/3.png)

즉, 두 학습된 특징이 선형적으로 상관이 없어야 한다.  그렇지 않다면 가장 느린 신호 하나만 포착하는 결과나 나올 수 있다.

## 13.4 Sparse Coding

![_config.yml]({{ site.baseurl }}/assets/ch13/4.png)

요약: 어떤 input data를 학습시킬 때 부분 특성들을 나타태는 파트들의 linear combination으로 나타낼 수 있는데, 이 대 곱해지는 latent variable이 대부분 0으로 이루어지게 한다.

우리에게 100장의 왜곡된 이미지들과 그에 상응하는 label로 이루어진 데이터를 가정해보자. 먼저 100장의 왜곡된 이미지들에서 품질과 관련되어 있는 이미지 특성들(f)을 도출한다. 20개를 도출했다고 하자. 즉, 각 이미지당 20개의 특성을 도출한 것이다. 한 이미지의 특성을 열 벡터(20 x 1)로 해서 행으로 차곡차곡 나열하면 하나의 행렬이 만들어지고, 이 행렬의 크기는 20 x 100이 된다. 이것을 사전(dictionary)이라고 부른다. 사전 안에 있는 하나의 열 벡터(column vector)들을 각각 atom이라고 부른다. 이 경우에는 사전 안에 100개의 atom이 있는 셈이다.

어떤 이미지의 품질을 평가하고 싶다면, 우선 마찬가지로 20개의 특성을 도출해야 한다. 그 다음에 그 특성을 사전에 있는 열 벡터들, 즉 atom들의 선형 조합으로 표현되게 한다.

$f=α_1f_1+α_2f_2+α_3f_3+⋯+α_{100}f_{100}$

![_config.yml]({{ site.baseurl }}/assets/ch13/5.png)

이를 위한 decoding 단계에서, 최대한 많은 계수들이 0이 되게 하는 Regularizer을 사용한다. 이것이 sparse coding method이다.

![_config.yml]({{ site.baseurl }}/assets/ch13/6.png)

sparse coding은 unsupervised learning의 한 방법으로, overcomplete basis vector을 기반으로 데이터를 효율적으로 표현하깅 위한 용도로 개발이 되었다.

이를 통해 원 데이터의 차원보다 큰 parameter space에서 더 밀도있는 표현을 가능하게 한다. sparse coding을 사용하면, data compression, noise제거, color interpolation 등에서 탁월한 성과를 얻을 수 있다.



## 13.5 Manifold Interpretation of PCA

- PCA나 factor analysis를 포함하는 linear factor model은 manifold 학습이라고 볼 수도 있음

  - Probabilistic PCA의 경우, 높은 확률을 가지는 얇은 팬케이크 모양의 영역을 정의하는 과정이라 할 수 있음

  - 이 때, Gaussian distribution은 팬케이크와 같이 한 축으로는 평평하고, 다른 축으로는 늘어난 모양을 같게 됨

  - 따라서 PCA는, 고차원 공간에서 linear manifold 형태의 이러한 팬케이크를 정렬하는 과정이라 할 수 있음

![_config.yml]({{ site.baseurl }}/assets/ch13/Fig_13.3.PNG)

- 이러한 해석은 전통적인 PCA 뿐만 아니라, $x$에 가깝도록 $x$를 재구성하겠다는 목표를 가지고 행렬 $W$, $V$를 학습시키는 어떠한 선형 오토인코더에도 적용될 수 있음

  - 저차원 representation $h$를 계산하는 다음의 인코더에 대해: $h = f(x) = W^T(x-\mu)$

  - 오토인코더의 측면에서는, 다음의 재구성을 하는 디코더를 갖게 됨: $\hat{x} = g(h) = b + Vh$

  - 이러한 선형 인코더, 디코더에 대해 재구성 오차 $\mathbb{E}[\mid\mid x-\hat{x}\mid\mid ^2]$ 를 최소화하는 조건은 $V=W$, $\mu=b=\mathbb{E}[x]$ 임

  - 이 때 $W$의 열들은, 공분산 행렬 $C = \mathbb{E}[(x-\mu)(x-\mu)^T]$ 의 주 고유벡터로 형성되는 부분 공간에 대한 orthonomal basis로 구성됨

  - PCA에서는 $W$의 열들이, 해당 고유값이 곱해진 고유벡터들로 구성됨


- Linear factor model은 가장 간단한 형태의 generative model 혹은 데이터의 representation을 학습하는 모델의 하나임

  - Linear classifier와 regression model이 deep feedforward network로 확장되듯이, linear factor model 또한 오토인코더나 deep probabilistic 모델로 확장되어 같은 과제를 더욱 강력하고, 유동적으로 처리하게 될 수 있음