Autoencoder는 자기 네트워크의 입력을 출력으로 복사하도록 학습하는 모델이다.
모델 구조는 크게 encoder $f$, decoder $g$로 나뉘며 입력 x에 대하여 $x=g(f(x))$이 모델을 찾는것이 목표이다.
보통 입력을 근사적으로 재현하지만, 이 과정에서 데이터셋의 가치있는 정보를 latent vector $h$에(책에서는 $h$를 "code"라고 부름) 압축할 수 있다.

물론 아무렇게나 하면 되는건 아니고 $h$에 유용한 정보를 압축하기 위해 네트워크의 모양이나 loss등을 잘 설계해야 한다.


## 14.1 Undercomplete Autoencoders
입력 $x$의 차원보다 code $h$의 차원이 작은 autoencoder를 undercomplete이라 부른다.
Undercomplete autoencoder에 대하여 기억해야 할 몇 가지 내용은 다음과 같다.

- 입력 차원보다 작은 code 차원으로 데이터를 압축해야 하기 때문에 데이터셋에서 유용한 정보만 추리도록 학습된다.
- 비용함수는 MSE를 사용한다.
    - $L(x, g(f(x)))$
- Decoder $g$가 선형 함수이고 MSE loss를 사용하는 경우 PCA와 같은 subspace를 학습한다.
    - 즉, undercomplete autoencoder는 PCA를 비선형 공간으로 일반화 시킨 것.

위의 특징들은 네트워크의 capacity가 적당히 작아 autoencoder가 데이터셋의 유용한 정보만 압축할 수 있다고 가정한 것이다.
만약 네트워크의 capacity가 너무 커서 입력을 완벽하게 복제할 수 있다면 autoencoder는 유용한 정보를 추출할 수 없다.


## 14.2 Regularized Autoencoders
위에서 언급한 네트워크 capacity 관련한 문제는 hidden code의 차원이 입력과 같거나 큰 경우 더 발생하기 쉽다.
예를들어 hidden code의 차원이 입력보다 더 큰 경우 (overcomplete) encoder와 decoder가 모두 linear여도 데이터셋의 유용한 정보를 학습하지 않고 입력을 완벽히 재현할 수 있다.

이상적으로는 데이터의 복잡도를 잘 고려하여 code의 차원을 잘 선택한다면, autoencoder를 잘 학습할 수 있다.
하지만, code의 길이를 제한하는 대신 autoencoder의 loss에 정칙화 항을 추가하여 데이터셋의 유용한 정보를 추출할수도 있다.
이런 방법을 regularized autoencoder라고 부른다.

### 14.2.1 Sparse Autoencoders
Sparse autoencoder는 loss에 code의 sparsity penalty $\Omega(h)$를 넣은 것이다.

$$
L(x, g(f(x))) + \Omega(h)
$$

#### Sparsity penalty 해석

보통 regularizer는 MAP 관점에서 prior로 해석된다.

1. MAP: maximizing $p(\theta | x)$
2. maximizing $p(\theta | x)$ <=> maximizing $\log p(x|\theta) + \log p(\theta)$

2번 식을 보면 prior는 parameter에만 의존해야 한다.
하지만 sparsity penalty $\Omega(h) = \Omega(f(x))$는 parameter와 데이터에 모두 의존하기 때문에 prior로 해석할 수 없다.

대신에 autoencoder는 latent variable을 가진 generative model의 MLE로 해석할 수 있다.
즉, 굉장히 큰 $h$값 하나를 선택하여 아래 식을 근사한다고 볼 수 있다.

$$
\log p_{model}(x) = \log \sum_h p_{model} (x, h)
$$

그렇다면 아래의 식을 최대화 하는 꼴이 된다.

$$
\log p_{model}(h, x) = \log p_{model}(h) + \log p_{model}(x|h)
$$

따라서 $p(h)$ 부분이 sparsity penalty에 해당한다.


### 14.2.2 Denoising Autoencoders
Sparse autoencoder처럼 penalty 항을 더하는 대신 loss 함수를 바꾸어 학습을 제어하는 방법도 있다.
Denoising autoencoder는 잡을을 더한 입력을 원본으로 복구하도록 학습된다.

$$
L(x, g(f(\tilde{x})))
$$

이 과정에서 denoising autoencoder는 잡을을 제외한 자료의 구조를 학습하게 된다.
따라서 데이터셋의 유용한 성질들을 학습할 수 있다.


### 14.2.3 Regularizing by Penalizing Derivatives
Sparse autoencoder와는 다르게 latent의 미분을 penalty로 사용하는 방법도 있다.

$$
L(x, g(f(x))) + \lambda \sum_i \vert\vert\nabla_xh_i\vert\vert^2
$$

이러한 penalty를 사용하면 입력의 조그만 변화에 대해 변하지 않는 latent를 얻게 된다.


## 14.3 Representational Power, Layer Size and Depth
많은 경우 하나의 hidden layer만 있는 autoencoder를 사용한다.
하지만 autoencoder도 순방향 신경망의 한 종류이기 때문에 층이 많아질수록 얻을 수 있는 이점이 있다.
- 보편근사정리에 따르면 hidden layer가 하나만 있어도 임의의 함수를 근사할 수 있다. 하지만, autoencoder의 관점에서 encoder와 decoder가 너무 얕으면 여러가지 제약을 적용하기 어렵다.
- 깊이가 깊어질수록 어떤 함수를 표현하기 위한 계산 비용이 지수적으로 줄어든다.
- 깊이가 깊어질수록 필요한 학습 데이터의 양이 지수적으로 줄어든다.
- 깊은 autoencoder가 더 나은 압축 성능을 보이는것이 실험적으로 증명되었다 (Hinton and Salakhutdinov, 2006).


## 14.4 Stochastic Encoders and Decoders
Autoencoder도 순방향 신경망의 한 종류이다.
따라서 순방향 신경망의 loss 함수를 설계했던 방법을 그대로 적용할 수 있다.
1. 순방향 신경망의 목표는 입력 $x$, 목표 출력 $y$에 대하여 $p(y \vert x)$를 최대화 하는 것.
    - 또는 $-\log p(y \vert x)$를 최소화
2. 모델의 목표에 따라 $p(y \vert x)$ 분포를 설정한 후 $-\log p(y \vert x)$를 loss로 사용하여 최소화.
    - 예를들어 출력이 연속적인 값이면 Gaussian을 선택 -> MSE loss

Autoencoder의 경우 입력과 목표 출력 모두 $x$인 경우인데, decoder 입장에서 다음과 같이 쓸 수 있다.
$$
\min[-\log p_{decoder}(x \vert h)]
$$
따라서, autoencoder에서도 모델의 목표에 따라 분포를 Gaussian, Bernoulli, ... 등으로 선택하면 loss 함수를 자연스레 유도할 수 있다.

Autoencoder가 확률론 관점에서 기존의 순방향 신경망과 다른 점은 encoder와 decoder의 확률 분포를 다음처럼 나누어 쓸 수 있다는 것이다.
$x$와 $h$의 어떤 결합 분포 $p_{model}(h, x)$에 대하여
- $p_{encoder}(h \vert x) = p_{model}(h \vert x)$
- $p_{decoder}(x \vert h) = p_{model}(x \vert h)$

단, 일반적으로 encoder와 decoder의 조건부 분포가 어떤 유일한 결합 분포 $p_{model}(h, x)$와 연결될 필요는 없다.
Denoising autoencoder의 경우 근사적으로 encoder와 decoder의 조건부 분포가 하나의 결함 분포와 연결되도록 학습되는 것이 밝혀졌다 (Alain et al; 2015).


## 14.5 Denoising Autoencoders

디노이징 오토인코더는 손상된 데이터를 입력으로 받아서 손상되지 않은 데이터를 출력으로 뱉는다. 데이터가 손상받는 과정을 $C(\tilde{\boldsymbol{\mathbf{x}}}\ \vert\  \boldsymbol{\mathbf{x}})$라 하면 디노이징 오토인코더는 아래와 같이 나타낼  수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch14/Fig14_3.png)

즉, 디노이징 오토인코더는 $(\tilde{\boldsymbol{x}},\boldsymbol{x})$를 통해 복원 분포  $$p_{\text{reconstrunct}}(\boldsymbol{x}\ \vert\ \tilde{\boldsymbol{x}}) = p_{\text{decoder}}(\boldsymbol{x}\ \vert\ \boldsymbol{h})$$를 학습하게 된다. ($\boldsymbol h = f(\tilde{\boldsymbol{x}})$) 

일반적으로 아래와 같은 기댓값에 대해 경사하강법을 적용하여 학습시킨다.

$$-\mathbb E_{\boldsymbol{\mathbf{x}}\sim \hat p_{\text{data}}(\boldsymbol{\mathbf{x}})} \mathbb E_{\tilde{\boldsymbol{\mathbf{x}}}\sim C(\tilde{\boldsymbol{\mathbf{x}}}|\boldsymbol{x})} \log p_{\text{decoder}}(\boldsymbol x\ |\ \boldsymbol h=f(\tilde{\boldsymbol{x}}))$$

### 14.5.1 Estimating the Score

모델이 모든 학습 포인트 $\boldsymbol{x}$에서 데이터 분포와 동일한 점수를 갖도록 장려함으로써 확률 분포의 일관된 추정치를 제공하는 '점수 일치(Score matching)'는 maximum likelihood의 대안이 될 수 있다. 여기서 점수는 기울기장 $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$이다. '점수 일치'는 18단원에서 자세히 다룰 예정이고, 이번 단원에서는 '기울기장을 학습하는 것 또한 $p_{\text{data}}$를 학습하는 하나의 방법이다' 정도만 알아도 충분하다.

![_config.yml]({{ site.baseurl }}/assets/ch14/Fig14_4.png)

빨간 X표를 데이터 $\boldsymbol x$, 회색 화살표를 $C(\tilde{\boldsymbol{x}} \vert \boldsymbol{x})$ 그리고 검은 실선(manifold) 주변으로 데이터 $\boldsymbol{x}$가 모여있다고 하자. 

- 디노이즈 오토인코더를 $\vert\vert g(f(\tilde{\boldsymbol{x}}))-\boldsymbol{x}\vert\vert^2$을 최소화 하도록 학습 시킨다면 $g(f(\tilde{\boldsymbol{x}}))$은 데이터 $\boldsymbol x$의 무게중심에 가까워진다.
- 즉, manifold 외부의 점으로부터 manifold로 이동하는 벡터(초록색) $g(f(\boldsymbol x))-\boldsymbol x$ 를 학습하는 것으로 생각할수있다.
- 또한 데이터 $\boldsymbol x$의 분포가 가우시안이라면(: $p_{\text{data}}(\boldsymbol x) \sim e^{-\vert\vert \boldsymbol x-\text{c.m}\vert\vert^2}$) 앞에서 정의한 '점수'는 $\nabla_{\boldsymbol x}\log p_{\text{data}} \sim \left(\text{C.M(manifold)}-\boldsymbol x\right)$가 되어서 벡터 ($g(f(\boldsymbol x))-\boldsymbol x$)와 동일한 의미를 가지게 된다.

따라서 가우시안 노이즈를 가지는 데이터에 대해  $\vert\vert g(f(\tilde{\boldsymbol{x}}))-\boldsymbol x\vert\vert^2$ 를 최소화 하도록 오토인코더를 학습시키는 것은 '점수'를 추정하는 과정과 같다. 이렇게 학습된 벡터장을 그려보면 아래와 같다.

![_config.yml]({{ site.baseurl }}/assets/ch14/Fig14_5.png)


## 14.6 Learning Manifolds with Autoencoders

- 다른 기계 학습 알고리즘들과 같이, 오토인코더도 데이터를 저차원 혹은 작은 크기의 manifold로 변환함


- Manifold의 중요한 성질로, 접평면(tangent plane)의 집합이 있음

  - 접평면은 $d$차원 manifold의 한 점 $x$에서, manifold가 유지되는 범위에서 변화가 가능한 basis vector들로 정의됨

  - 따라서 해당 평면에서 $x$가 약간 변했을 때, manifold는 유지됨


![_config.yml]({{ site.baseurl }}/assets/ch14/Fig_14.6.PNG)


- 오토인코더의 학습 과정은 두 성질의 절충으로 이루어짐

  1. 학습 데이터 $x$의 representation $h$를 학습하여, 디코더에 의해 $h$로부터 $x$가 복원되도록 함

      - $x$가 학습 데이터에서 추출되었다는 점은, 데이터 분포에 존재하지 않는 인풋은 고려하지 않아도 된다는 점에서 중요한 성질임

  2. 정규화의 페널티를 만족하도록 함
  
      - 오토인코더의 용량을 제한하는 구조적인 한계, 재구성 비용에 더해진 정규화 텀 등을 이용해 인풋 변화에 덜 예민한 솔루션을 유도함.

- 둘 중에 한 조건만 충족하는 것으로는 의미가 없음

  - 단순히 인풋을 아웃풋으로 복사하는 것으로는 쓸모가 없음

  - 두 성질이 함께 작용하여 hidden representation이 정보를 추출하도록 하기에 의미가 있음

  - 따라서 오토인코더는 학습 데이터를 재구성하는데 필요한 정보만 학습한다는 점이 중요한 성질임

  - 이로 인해 재구성하는 지점 근처에서 인풋의 변화에 덜 예민한 재구성 함수가 얻어짐

![_config.yml]({{ site.baseurl }}/assets/ch14/Fig_14.7.PNG)


- 오토인코더가 manifold 학습을 위해 왜 유용한지 이해하기 위해서는, 다른 알고리즘들과 비교해볼 수 있음

  - 일반적인 알고리즘에서는 모든 학습 데이터마다 개별적으로 respresentation을 추출함

  - 하지만 오토인코더에서는 저차원의 manifold가 유지되는 ambient space(input space)의 모든 지점에 대한 mapping을 추출함


- 비선형 manifold를 학습하는 초기 연구들에서는 nearest-neighbor graph에 기반한 non-parametric 기법에 초점을 두어 옴

  - 학습 데이터의 한 지점을 기준으로, manifold상 가장 가까운 지점들을 연결하여 접평면을 구성함

![_config.yml]({{ site.baseurl }}/assets/ch14/Fig_14.8.PNG)


- 선형 시스템을 최적화함으로써, 여러 Gaussian 패치가 타일링 되어 이루어진 manifold를 계산할 수도 있음

  - 각각의 패치는 수직 방향으로는 변화가 작고, 좌표계 방향으로는 변화가 큰 팬케이크 모양으로 정의할 수 있음


![_config.yml]({{ site.baseurl }}/assets/ch14/Fig_14.9.PNG)


- 하지만 manifold가 부드럽게 변하지 않는 다면, local 기반 접근으로 문제를 해결하기 어려움

  - 학습 데이터가 매우 많다면 해결 될 수 있지만, AI 분야에서 다루는 문제의 복잡성을 고려할 때 쉽지 않음
  
  - 이로 인해 manifold의 구조 자체를 파악하는 방법에 대한 연구들이 관심을 받고 있음

## 14.7 Contractive Autoencoders(CAE)

Contractive Autoencoders(축약 자동부호기)는 부호 $h=f(x)$에 아래와 같은 명시적인 regularization term을 추가한다.

![_config.yml]({{ site.baseurl }}/assets/ch14/1.png)

penalty $\Omega (h)$는 Jacobian matrix의 편미분의 squared Frobenius norm으로 만들어져있다. denoising autoencoder와 유사점이 있는데, denoising autoencoder가 작은 Gaussian input noise의 한계 안에서 autoencoder의 reconstruction error과 Contractive Autoencoder의 Contractive panelty항이 동등한 역할을 한다.

denoising autoencoder가 작은 유한한 크기의 perturbation에 저항하도록 설계되었다면 Contractive autoencoder은 feature extraction function이 무한소의 perturbation에 저항하도록 만든다.

'축약'이라는 표현이 사용되는 이유는, CAE가 input perturbation에 저항하도록 훈련되므로 training point x는  그 부근의 point들과 출력 점들의 더 작은 이웃 영역으로 mapping되도록 유도된다. 다시 말해 input 영역이 축약된다고 할 수 있다.

CAE를 사용하면 대부분의 점에서 ${\partial f(x) \over \partial x}$가 작은 값이 되는 autoencoder이 나온다.

![_config.yml]({{ site.baseurl }}/assets/ch14/2.png)

## 14.8 Predictive Sparse Decomposition(PSD)

sparse coding과 parametric autoencoder의 짬뽕이다. PSD는 반복적 추론의 결과를 **예측**하도록 parametric autoencoder을 훈련한다. PSD는 encoder $f(x)$와 decoder $g(h)$로 구성되어 있다. 학습은 아래를 최소화하는 것으로 진행된다.

![_config.yml]({{ site.baseurl }}/assets/ch14/3.png)

sparse coding에서처럼 훈련 과정은 h에 대한 최소화, model parameter에 대한 최소화를 번갈아 진행한다.

PSD는 learned approximate inference의 한 예이다.

## 14.9 Applications of Autoencoders

차원축소에 autoencoder가 많이 사용되고 있다.

Autoencoder을 사용한 차원 축소가 가장 잘 먹히는 곳으로 information retrieval(정보 조회)가 있다. 주어진 query entry와 유사한 항목들을 데이터베이스에서 찾아내는 것을 말한다. 저차원 binary parameter을 산출하도록 dimension reduction algorithm을 학습하면, DB의 모든 항목을 하나의 hash table에 저장할 수 있다. (0111011011010) 이 해시 테이블을 이용해 주어진 query와 동일한 binary code를 리턴함으로서 정보 조회가 끝난다. 이를 sementic hashing이라고 한다. binary code로 어떻게 만들까? 마지막 layer에 sigmoids를 encoding function과 함께 배치하면 된다.