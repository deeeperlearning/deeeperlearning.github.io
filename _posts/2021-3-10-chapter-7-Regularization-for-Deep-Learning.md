ML의 주요 문제는, 알고리즘이 training set 뿐만 아니라 새로운 input에 대해서도 잘 작동하게 만드는 것이다. ML의 training session에서 많이 쓰이는 전략 중에는, training set의 정확도를 희생해서라도 test test에 대한 정확도를 높이는 것들이 있다. 이런 전략들을 regulariation이라 통칭한다.

Regularization 전략은 다양하다. 모형에 추가 제약을 거는 방법이 있고, objective function에 항을 추가해 softness에 대한 constraint를 더한다. 어떤 regularization 방법은 prior knowledge에 기대기도 한다.

대부분의 regularization strategy는 regularizing estimator에 기초한다. 어떤 estimator의 regularization은 bias 증가와 variance 감소를 trade off하는 방식으로 이주어진다. 따라서 좋은 regularization은 bias를 최대한 적게 높이면서 variance를 최대한 많이 높이는 것이다.

5장에서는 generalization, over fitting에 대한 세가지 상황을 이야기했다.

- true data generating process가 제외된 상황(underfitting+bias)
- model에  the true data generating process가 포함되어있는 상황
- model에 true data generating process뿐만아니라 다른 생성 process도 섞여있는 상황(bias가 아니라  variance가 error에 크게 기여하는 상황→overfitting)

Regularization의 목표는 세 번째 상황을 두 번째 상황으로 바꾸는 것이다. However, most applications of deep learning algorithms are to domains where the true data generating process is almost certainly outside the model family.

# 7.1 Parameter Norm Penalties

많은 regularization 방법들은 최적화 알고리즘의 목적함수에 어떠한 값을 추가해 모델의 수용력을 원하는 범위로 제한한다. 즉, 아래처럼 원래 목적함수 $J(\theta; X, y)$에 penalty $\Omega(\theta)$를 더하는 식이다.

$$\tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha \Omega(\theta)$$

$\alpha$는 penalty의 강도를 조절하는 실수이다 ($\alpha \in [0, \infty]$). 

이제 최적화 알고리즘은 $\tilde{J}$를 최적화하여 모델을 결정한다. 따라서, $\Omega$를 어떤 형태로 선택하는지에 따라 학습된 모델의 형태가 달라진다.

이 절에서는 신경망의 가중치들에만 penalty를 가한 경우를 고려할 예정이다. 신경망의 파라미터 중 bias를 제외한 가중치들에만 penalty를 가하는 것이 일반적인데, 이유는 다음과 같다.

- Bias들은 더 적은 양의 training set으로도 optimal한 값을 찾을 수 있다.
- Bias들은 penalty 없이 학습해도 분산이 (가중치에 비해 상대적으로) 커지지 않는다.
- Bias들에 penalty를 적용할 경우 모델이 과소적합 될 가능성이 크다.

따라서 이후의 논의들에 등장하는 $w$는 penalty를 적용하는 가중치들을 뜻하고 $\theta$는 $w$를 포함한 모든 파라미터를 뜻한다.

## 7.1.1 $L^2$ Parameter Regularization

가장 많이 사용되는 $L^2$ norm penalty이다. 이 경우 $\Omega = \frac{\alpha}{2}w^Tw$이다. 주로 weight decay, ridge regression, Tikhonov regulariation이라 불린다. 단순화하기위해 bias 항은 없다고 가정하면 전체 목적함수는 아래와 같다.

$$\tilde{J}(w; X, y) = J(w; X, y) + \frac{\alpha}{2} w^Tw$$

### 가중치 감쇄 효과

$L^2$ Parameter Regularization은 주로 weight decay라고 표기된다. 기울기 기반 최적화 알고리즘을 이용해 $\tilde{J}$를 최적화하는 과정에서 $\frac{\alpha}{2} w^Tw$ 항이 어떻게 작용하는지 살펴보면 이러한 이름이 붙은 이유를 알 수 있다. 한번의 iteration 동안 최적화 알고리즘이 계산하는 값을 살펴보면

$$\nabla_w \tilde{J}(w; X, y) = \nabla_w J(w; X, y) + \alpha w$$

즉,

$$w \leftarrow w - \epsilon(aw + \nabla_wJ(w;X,y)) = (1-\epsilon\alpha)w - \epsilon\nabla_wJ(w;X,y)$$

위 식의 우변을 보면 $\nabla_wJ(w;X,y)$에 상관없이 매 iteration마다 가중치들이 일정 비율로 작아지는 것을 알 수 있다.

### Optimal point 근처에서 2차 근사를 이용한 해석

$L^2$ Regularization이 어떻게 작동하는지 알아보기 위해 optimal point 근처에서 목적함수를 2차함수로 근사해보자. $J$만 최적화 시킨 경우의 optimal point를 $w^* = \arg\min_wJ(w)$ 라고 표시하고 이 근처에서 $J$를 2차근사하면

$$\hat{J}(w) = J(w^*) + \frac{1}{2} (w-w^*)^TH(w-w^*)$$

여기에서 $H$는 $w=w^*$인 점에서 $J$의 Hessian matrix이다. 이제 가중치 감쇄 항을 추가한 목적함수를 최적화하기 위해 $\hat{J}$에 $\frac{\alpha}{2}w^Tw$를 더한 후 미분하여 최종 해 $\tilde{w}$를 찾아보면

$$\nabla_w(\hat{J}(w)+\frac{\alpha}{2}w^Tw) = H(w-w^*) + \alpha w$$

$\tilde{w}$에서 기울기가 0이 되어야 하므로

$$H(\tilde{w}-w^*) + \alpha \tilde{w} = 0$$

$$\tilde{w} = (H+\alpha I) ^{-1} H w^*$$

해석하기 편하게 $H$를 eigen decomposition 하면 ($H = Q\lambda Q^T$)

$$\tilde{w} = Q(\Lambda+\alpha I) ^{-1} \Lambda Q w^*$$

즉, $w^*$의 성분 중 $i$번째 고유벡터 ($Q_{:, i}$) 방향의 성분들은 $\frac{\lambda_i}{\lambda_i + \alpha}$만큼 rescaling 된다. 따라서 $J$의 Hessian matrix의 principal 방향 중 기울기가 큰 방향으로는 ($\lambda_i \gg \alpha$) $J$를 최적화 하기 위한 최적의 파라미터 $w$가 선택되고, 기울기가 작은 방향으로는 ($\lambda_i \ll \alpha$) 0에 가까운 $w$가 선택된다. 아래 그림을 보면 가로축으로는 0에 가깝고 세로축으로는 $J$를 최적화하는 $w$가 선택된 것을 볼 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch7/norm_fig_1.png)

### 선형 회귀 관점에서 가중치 감쇄의 효과

선형 회귀에 가중치 감쇄를 적용하면 목적함수가

$$(Xw-y)^T(Xw-y)$$

에서

$$(Xw-y)^T(Xw-y) + \frac{\alpha}{2}w^Tw$$

로 변한다. 두 경우 모두 closed form solution이 존재하는데, 원래 목적함수와 가중치 감쇄가 추가된 목적함수의 해는 순서대로 아래와 같다.

$$w^* = (X^TX)^{-1}X^Ty$$

$$w^* = (X^TX+\alpha I)^{-1}X^Ty$$

즉, 학습 알고리즘이 데이터 $X$의 각 feature의 분산을 더 크게 느끼게 만드는 효과가 있다.

## 7.1.2 $L^1$ Regularization

$L^1$ regularization은 $\Omega = \alpha \sum_i|w_i|$인 경우이다. 전체 목적함수는 다음과 같다.

$$\tilde{J}(w; X, y) = J(w; X, y) + \alpha |w_i|$$

목적함수의 기울기를 살펴보면

$$\nabla_w \tilde{J}(w; X, y) = \nabla_w J(w; X, y) + \alpha \text{sign}(w)$$

인데, $L^2$와는 다르게 기울기가 가중치의 값에 상관없이 부호의 영향만 받는다.

### $L^1$ 정칙화 해석 - 식의 관점에서

7.1.1과 동일하게 $J$의 해($w^*$) 근처에서 목적함수를 2차근사 한 후 식을 살펴볼 예정이다. 단, 논의를 단순화하기 위해 $w^*$ 근처에서 $J$의 Hessian은 대각행렬이라고 가정한다(이러한 가정이 어느정도 타당한 이유는 PCA를 이용해 전처리하면 서로 다른 차원들 사이의 의존성을 없앨 수 있기 때문). $w^*$ 근처에서 $J$를 2차근사하고 penalty항을 더하면

$$\hat{J}(w) = J(w^*) + \frac{1}{2} (w-w^*)^TH(w-w^*) + \sum_i\alpha|w_i| \\ = J(w^*) + \sum_i[\frac{1}{2}H_{i, i}(w_i-w_i^*)^2] + \sum_i\alpha|w_i|$$

이 경우에 해석적 해가 존재하는데, 다음과 같다.

$$\tilde{w} = \text{sign}(w_i^*)\max\{{|w_i^*|-\frac{\alpha}{H_{i, i}}, 0}\}$$

$w^*_i>0$인 경우만 생각해보면

1. $w_i^*<\frac{\alpha}{H_{i, i}}$인 경우 $\tilde{w} = 0$
2. $w_i^*\geq\frac{\alpha}{H_{i, i}}$인 경우 $\tilde{w} >0$인 optimal value

즉, $L^2$ regularization과 비슷하게 Hessian의 경사가 가파른 방향의 가중치는 $J$를 최적화 하는 방향으로 움직이고 상대적으로 경사가 완만한 방향에서는 가중치를 0으로 선택한다. 이해를 돕기 위해 아래 그림을 보자.

### $L^1$ 정칙화 해석 - visual inspection

![_config.yml]({{ site.baseurl }}/assets/ch7/norm_fig_2.png)

위에서 본 케이스는 Hessian이 대각행렬인 경우이다. 이 경우 해가 단순해지는 이유는 Hessian의 등고선이 $L^1$ penalty 등고선의 꼭지점에 접할 확률이 높기 때문 (case1, 2).

Hessian이 대각행렬이 아닌 경우 sparse한 해를 가질 확률이 적어진다 (case3). 물론 이 경우에도 sparse한 해를 얻을 가능성이 있다 (case4).

# 7.2 Norm Penalties as Constrained Optimization

## Constraint optimization 관점에서 정칙화 해석

4.4에서는 라그랑주 함수와 KKT 컨디션을 이용하여 constrained optimization 하는 방법을 다루었다. 이 관점에서 정칙화가 어떤 의미를 가지는지 살펴보자. 우선 정칙화 항이 포함된 비용함수는 다음과 같다.

$$\tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha \Omega(\theta)$$

만약 여기에 $\Omega$가 $k$보다 작아야 한다는 제약을 추가하면 최적화해야하는 라그랑주 함수와 해는 다음과 같이 적힌다.

$$L(\theta, \alpha; X, y) = J(\theta; X, y) + \alpha(\Omega(\theta) - k)$$

$$\theta^* = \arg\min_{\theta} \max_{\alpha, \alpha \geq 0} L(\theta, \alpha)$$

논의를 단순화하기 위해 (그리고 실제로 딥러닝에서 하듯이) $\alpha = \alpha^*$로 고정하고 $\theta$에 대한 최적화만 생각하면

$$\theta^* = \arg\min_{\theta} (J(\theta; X, y) + \alpha^* \Omega(\theta))$$

즉, 우리가 regularization term을 추가한 비용함수를 최적화하는 것은 $\Omega$가 $k$보다 작아야한다는 제약을 추가한 경우의 문제를 푸는 것과 같다. 단, 우리는 $\alpha$를 임의로 정한 후 고정하였으므로 다른 점들도 있다.

- 라그랑주 함수를 최적화하는 경우 $\alpha$도 최적화되지만 우리는 그냥 정한다. Grid search등을 이용하여 $\alpha$를 간접적으로 최적화하기도 한다.
- 실제 라그랑주 함수 최적화에서는 $k$가 정해지면 이에 따라 최적화의 결과로 $\alpha^*$가 정해진다. 하지만 $\alpha$를 특정 값으로 정했을 때 $k$가 몇에 해당하는지는 알 수 없다. 하지만 $\alpha$를 키울수록 $k$는 작은 경우에 해당한다. 즉 더 강하게 정칙화할수록 가중치들이 0에 가까워진다.

## 명시적인 제약을 가하는 방법

가끔은 우리가 학습시킨 모델의 가중치들이 정말로 특정 값보다 작게 만들고 싶을 수 있다. 이 경우 4.4에서 소개한 방법을 사용할 수 있다.

매 학습 iteration마다

1. $J$의 미분값을 이용하여 $w$ 업데이트
2. 1의 결과로 찾아진 $w$에 대하여 명시적인 제약을 만족하는 가장 가까운 점으로 $w$를 업데이트

이렇게 하면 특정 제약값보다 항상 작은 $w$를 얻을 수 있다.

이러한 방법을 사용하면 몇 가지 이점이 있는데,

- 이미 제약 값보다 작아진 가중치를 더 작게 만들지 않는다. 따라서 너무 많은 가중치들이 죽어버리는 (0으로 고정되어버리는) 일이 발생하지 않는다.
- 가중치가 발산해버리는 것을 막을 수 있다. 정칙화에 대하여도 기울기 기반으로 최적화 할 경우 **기울기 너무 큼 → 가중치 너무 커짐 → 기울기 더 커짐** 의 발산이 발생할 수 있지만 명시적인 제약을 사용하면 이런 일을 막을 수 있다.

# 7.3 Regularization and Under-Constrained Problems

기계학습으로 under-constrained 문제를 풀어야 할 경우 정칙화가 꼭 필요할 수 있다. 몇 가지 예시를 들자면

- PCA의 covariance matrix $X^TX$가 singular하여 해를 구할 수 없는 경우 → covariance matrix를 $X^T X + \alpha I$로 정칙화하여 해결 가능
- 선형 회귀 문제의 데이터가 적어 under-constrained인 경우 $w$로 데이터 분류가 가능하다면 $2w$로도 분류 가능. 즉, $w$가 발산할 가능성이 있음 → 정칙화를 통해 방지 가능

# 7.4 Dataset Augmentation

기계학습 모델의 일반화를 돕기 위해서는 큰 데이터셋이 필요하다. 하지만 우리가 가진 데이터는 한정되어 있다. 이 경우 비용함수에 정칙화 항을 추가하는 것 말고도 데이터셋 자체를 수정하여 정칙화를 하는 것이 가능하다. Data augmentation의 기본 발상은 **"실제로 있을법한 변환을 가지고있는 데이터셋에 적용하여 데이터셋 사이즈를 키우는 것"** 이다.

- Image classification을 하는 경우 이미지를 회전하거나 translation 하는 변환은 현실에서 있을법 하다. 이러한 변환을 통해 데이터셋의 사이즈를 키울 수 있다.
- 음성 인직 모델에서 데이터셋에 noise를 추가하여 데이터셋의 사이즈를 키울 수 있다.

이러한 augmentation을 하면 augmentation에 사용된 연산의 종류에 대하여 모델이 robust하게 학습된다. 단, 변환하는 규칙은 잘 선택해야 한다. 예를들어 MNIST 분류 문제에서 180도 회전하는 연산을 이용해 augmentation 하게 되면 모델이 9와 6을 구분하지 못하게 될 수 있다.




## 7.5 Noise Robustness

일부 model에서는 infinitesimal variance를 가진 noise를 training input에 추가하는 것과 weights의 norm으로 regularize하는 것이 동등하다. (Bishop 1995a, b)

일반적인 경우 잡음 주입이 그냥 매개변수 weight 크기를 줄이는 것보다 효과적일 수 있다. 

잡음을 input이 아니라 weight에 더하는 방법도 있다. (Graves, 2011) 베이즈 통계 관점에서 이는 불확실성을 확률적으로 반영하는 방법이다. 

아래 MSE를 사용하는 l층 MLP 문제가 있을 때 weight에 perturbation $\epsilon_W \sim N(\epsilon;0, \eta I)$ 을 가한다고 해보자. 이 모형을 $\hat{y}_{\epsilon_W}(x)$로 표기한다. 이 문제의 objective function은 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%204.png)

작은 $\eta$에 대해 잡음이 추가된 objective function을 최소화하는 것은, 아래와 같은 regularization term을 추가한 objective function 최적화와 동등하다. → 왜죠?!

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%205.png)

이 정칙화는 parameter들을 가중치의 작은 perturbation이 출력에 비해 작게 영향을 미치는 여역들로 이끈다.(평평한 영역들에 둘러싸인 극소점들에 도달)

### 7.5.1 Injecting Noise at the Output targets

label y가 잘못 부여된 data가 있을 수도 있다. 이런 input이 들어가면 $log p(y \mid x)$의 예측 성능에 나쁜 영향을 준다. 이를 방지하는 한 방법은 label에 명시적인 noise를 추가하는 것이다. $\epsilon$이 작은 constant라고 할 때, y가 정확할 확률을 $1-\epsilon$으로 두는 것이다. 이런 방식을 label smoothing이라고 한다. label smoothing은 pursuit of hard probabilities without discouraging correct classification. 지금도 많이 쓰인다. 

## 7.6 Semi-supervised Learning

semi-supervised learning은 P(x)에서 추출한 unlabeled example과 p(x, y)에서 추출한 labeled example을 동시에 사용한다. 심층학습의 맥락에서 semi-supervised learning은 $h = f(x)$를 배우는 것을 말할 때가 많다. 같은 부류에 속하는 example들이 비슷하게 표현되는 하나의 representation을 배우는 것이다. 

두 부류의 데이터에 대해 unsupervised learning의 구성요소와 supervised learning의 구성요소를 따로 두는 대신 P(x), p(x, y)의 generative model이 p(y|x)의 판별 모형(discriminative model)과 parameter을 공유하는 모형도 만들 수 있다. 이 경우 supervised learning의 판정 기준 -logP(y|x)와 unsupervised learning 의 학습, 생성 판정 기준을 절충할 수 있다. 

## 7.7 multitask learning

multitask learning은 여러 과제에서 발생한 example들을 저장, 공유해서 모델의 일반화 정도를 개선하는 방법이다. data가 많으면 paramter들의 generalize가 더 잘 되는 것과 마찬가지로 여러 과제가 모형의 한 부분을 공유하면 그 부분의 일반화가 개선될 수 있다. 아래 그림이 대표적이 예이다. 서로 다른 지도학습 과제들이 동일 입력 x를 공유하며, 일부 중간 수준 표현 h도 공유하여 일부 공통 요인들의 pool을 형성한다.!

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%206.png)

일반적으로 이런 multitask learning은 두 부분, 그리고 이와 관련된 parameter로 구성된다. 

1. Task-specific parameters (which only benefit from the examples of their task
to achieve good generalization). These are the upper layers of the neural
network in Fig. 7.2.

2. Generic parameters, shared across all the tasks (which benefit from the
pooled data of all the tasks). These are the lower layers of the neural network
in Fig. 7.2.

매개변수들을 공유하면, statistical streangth가 크게 개선될 수 있다. 결과적으로 일반화, 일반화 오차 한계들이 개선될 수 있다. 이는 서로 다른 과제들 사이의 통계적 관계가 유효하다는 가정이 성립할 때의 이야기다. 

심층 학습의 관점에서 이러한 접근 방식에 깔린 사전 믿음은 서로 다른 과제들에 연관된 자료에서 관측된 변동들을 설명하는 요인들 중 일부를 둘 이상의 과제들이 공유한다는 것이다. 

## 7.8 Early Stopping

높은 capacity를 가진 model을 훈련할 때 epoch가 반복됨에 따라 training error가 줄어들면서 validation error가 증가하는 경우가 있다. 아래처럼. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%207.png)

이를 해결하기 위해 최적의 validation error을 가질 때의 parameter을 저장해두었다가 이를 최종 결과로 보내는 방법이 있으며 이를 early stopping이라고 한다. Deep Learning에서 가장 흔히 쓰이는 Regularization 방법중 하나다. 

Early stopping이라는 Regularization method에서 Epoch 개수가 곧 hyperparameter가 된다. 이 hyperparameter의 특징은, learning process에서 여러 hyperparameter을 여러개 시도해본다는 것이다. 

Early stopping은 a validation set이 필요하다. 그만큼 training set의 크기가 작아진다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%208.png)

- 왜 early stopping이 Regularization인가?

from <Bishop 1995a>, 조기 종료가 최적화 절차를 매개변수 공간에서 초기 매개변수값 $\theta_o$ 비교적 작은 영역으로 한정하는 효과를 낸다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%209.png)

왼쪽 그래프에서 실선은 negative log likelihood이고, 점선은 SGD의 자취다. early stoping은 최적 지점 $w^*$까지 도달하지 않고 그 전에서 멈춘다. 오른은 비교를 위한 L2 norm Regularization 그래프다. 

learning rate가 $\epsilon$이고, optimization을 $\gamma$번 수행한다고 할 때, $\epsilon\gamma$는 effective capacity를 나타내는 하나의 값이라고 할 수 있다. 학습속도를 제한하면, 위 그래프에서 $w^*$에 도달하기 전에 멈춰버리니까!

quadratic error function, simple gradient descent를 사용하는 경우 early stopping과 $L^2$ norm regulariation이 동등함을 증명할 수 있다. 
optimal point $w^*$ 근처의 cost function은 아래와같이 근사할 수 있다 . (talor series)

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2010.png)

이 cost function의 w에 대한 기울기는 아래와 같다. 

$$\nabla_w \hat{J}(w) = H(w - w^*)$$

이 경사를 따라서 w가 update될 때를 식으로 표현하면, 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2011.png)

H를 eigen-decomposition하면($H = Q\Lambda Q^T)$ r결국 아래와 같은 식이 도출된다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2012.png)

앞의 L2 norm 정칙화 설명에서 아래와 같은 식이 나오는데, 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2013.png)

결론적으로 아래와 같은 조건을 만족하면 early stopping과 L2 regularization이 같은 뜻이 된다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2014.png)

## 7.9 Parameter Tying and Parameter Sharing

parameter의 적절한 값이 무엇인지는 모르지만 parameter 사이의 어떤 의존성들이 존재한다는 것을 추론할 수 있을 때 Parameter Tying and Parameter Sharing을 사용한다. 

의존성의 대표적인 예로, 특정 parameter들의 서로 가까워야 한다는 가정이 있다. 비슷한 과제에 대한 모형 A와 B의  parameter을 각각 $w^{(A)}$와 $w^{(B)}$라고 하자. 두 모형은 연관된 출력으로 입력을 mapping하는 모형들이다. 

$$\hat{y}^{(A)} = f(w^{(A)}, x), \ \hat{y}^{(B)} = f(w^{(B)}, x)$$

이때, 두 과제가 비슷하니 parameter들도 서로 비슷하다는 사전믿음( $w_i^{(A)} \sim w_i^{(B)}$)이 있다고 가정하자. 이런 사전 믿음을 regularization을 통해 활용할 수 있다. 즉 아래와 같은 regularization term을 objective function에 추가할 수 있다. 

$$\Omega(w^{(A)}, w^{(B)}) = \Vert w^{(A)} - w^{(B)} \Vert^2_2$$

parameter들이 서로 같아야한다는 제약을 적용하는 정칙화를 parameter sharing이라고 한다. 

### 7.9.1 Convolutional Neural Network

이미지 데이터에서는 translation에 불변힌 통계적 성질들이 있다. 예를 들어 고양이 사진의 한 스코프에서 몇 픽셀을 옮겨도 고양이 사진이다. CNN은 이런 성질에 근거해 이미지의 여러 위치에서 parameter을 공유한다. 이 제약을 통해 model parameter가 크게 줄어들며 모델의 전체 사이즈를 늘리기 않으면서 더 깊게 만들 수 있다. 

## 7.10 sparse representation

weight decay는 model parameter에 직접 panelty를 가하는 형태로 수행된다. 이렇게 하는 대신, activation of units에 panelty를 가하는 regularization 전략도 있다. $L^1$ norm panelty는 parameter 분포를 sparse하게 만드는데, 이는 다수의 parameter가 0에 가까운 값이 된다는 뜻이다.

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2015.png)

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2016.png)

아래쪽 선형 모형이 희소 표현 h를 보여준다. x의 함수인 h는 x에 대한 정보를 아래쪽 모형에서 더 희소하게 표현한 것이다. 

represenatation에 대한 norm qanelty regularization은 loss function J에 이에 대한 항 $\Omega(h)$을 추가해 수행한다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2017.png)

parameter에 대한 regularization이 parameter의 sparsity를 만든다면, representation의 element에 대한 regularization은 element의 sparsity를 만들어낸다. 

activation value에 강한 constraint를 가해 sparsity를 만드는 방법도 있다. orthogonal matching pursuit은 input x를 아래와 같은 optimization problem을 푸는 represenatation h로 부호화한다. 

![_config.yml]({{ site.baseurl }}/assets/ch7/Untitled%2018.png)



## 7.11 Bagging and Other Ensemble Methods

- Bagging: 여러 모델을 합쳐 generalization error를 줄이는 방식
- 여러 모델을 독립적으로 학습시키고, 각각의 모델이 text example에 대해 결정을 내림 (vote)
- Model averaging 혹은 ensemble methods이라 불림
- 작동 이유: 서로 다른 모델이 보통 test set에 대해 모두 같은 에러를 만들지는 않기 때문
- 예) k개 regression model이 있고, 모델 $i$의 에러를 $\epsilon_i$라 할 때,
	- variance $E[\epsilon_i^2] = v$, covariance $E[\epsilon_i\epsilon_j] = c$라 하자
	- 모든 model의 평균으로 얻어지는 에러는 $1/k\Sigma\epsilon_i$가 됨

![_config.yml]({{ site.baseurl }}/assets/ch7/Bagging.PNG)

- 모든 모델이 perfectly correlated 되어 $c = v$이면, 위 식의 값은 $v$가 되어, model averaging이 전혀 도움되지 않음
- 모든 모델이 perfectly uncorrelated되어 $c = 0$이면, 위 식은 only $v/k$가 됨
	- 따라서 ensemble은 최소한 각 모델의 성능만큼은 보일 수 있음
	- 각 모델의 error가 서로 independent 할수록 각 멤버보다 우수해짐
	- 예) 서로 다른 모델이 서로 다른 측면을 학습함

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_5.PNG)

### 7.11.1 Gradient Boosting

Boosting이라는 기법은 개별 모형보다 수용력이 더 큰 앙상블을 구축한다. bagging은 각 모형들이 독립적으로 결과를 예측하는데 반해, Boostting은 Residual fitting이라는 방식을 사용하여 영향을 주고받는다.

간략한 설명은 다음과 같다. 모델이 N개 있을 때, 첫번째 모델에 input X를 넣고, 예측이 잘못된 값들의 오차 방향(negative gradient)을 가중치에 더해 다음 모델에 넣는다. 이 과정을 총 모델의 개수만큼 반복한다. 그림으로 설명하면 아래와 같다.

![_config.yml](https://miro.medium.com/max/1400/1*DwvwMlOcT1T9hZwIJvMfng.png)

Bagging과 Boosting의 차이를 그림으로 나타내면 아래와 같다. 

![_config.yml](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbwr6JW%2FbtqygiHRbRk%2Fcy5hbDAPpTjCG7xa6UWxi0%2Fimg.png)


## 7.12 Dropout

- Dropout: 많은 수의 neural network로 이루어진 ensemble을 만드는 bagging 방식
- Bagging의 기본 원리는 각각의 test example에 대해 많은 수의 모델을 학습시키고, 평가하는 것임
- 하지만 각각의 model이 큰 네트워크라면, 이를 모두 학습/평가 하기 시간/메모리 측면에서 현실적이지 못함
- Dropout을 이용하면 효율적인 방식으로 여러 네트워크로 구성된 bagged ensemble을 만들 수 있음
- 일반적으로 non-output unit을 제거하는 방식으로 구현함
	- 혹은 구현의 편의를 위해 특정 unit의 output에 0을 곱함

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_6.PNG)

- 각 input/hidden unit이 model에 포함될 확률은 hyperparameter로, 학습 전에 정해짐
	- 일반적으로 input unit의 포함 확률은 0.8, hidden unit은 0.5를 사용

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_7.PNG)

- Inference: 모든 subnetwork member로부터 vote를 수집하는 과정
	- 일반적으로 10~20개의 sub-network (mask)만 이용해도 좋은 성능을 얻음


- Dropout의 일반적인 장점
	- Computationally cheap: per sample per update 마다 n개의 random binary number를 만들고 곱하는 O(n)의 연산만을 요구함
	- 제한 없음: 모델의 종류, 학습 과정에 구애받지 않음
	- 우수성: computationally inexpensive하다고 알려진 다른 regularizer보다 우수한 것으로 알려짐
		- weight decay, filter norm constraints, sparse activity regularization 등


- Dropout의 약점
	- Regularization이므로 모델의 capacity가 감소하기는 해서, 충분히 큰 모델을 사용해야 함
		- 이 때, dataset이나 network가 너무 크면 dropout에 의해 generalization error가 주는 것 보다, 연산량이 늘어나는 효과가 더 클수도 있음
	- Labeled training example의 수가 적을 때는 효과적이지 못한 것으로 알려짐
		- 이러한 조건에선 Bayesian neural network, unsupervised feature learning 등이 더 우수한 것으로 알려짐


- Dropout에 대한 이외의 이야깃거리
	- Linear regression에 적용할 땐, 각 input feature마다 다른 coefficient가 적용되는 $L^2$ weight decay와 동치임이 증명됨
	- 유성 생식에서 서로 다른 개체의 유전자가 조합되며 적응성을 높이듯이, 각 hidden unit들은 다른 특정 hidden unit이 모델에 포함되는지와 상관 없이 잘 작동해야 함
		- 모델끼리 hidden unit을 바꿔도 정상적으로 작동해야 함
	- Dropout의 장점은 input의 raw value의 noise라기보다, input의 information의 noise에 대한 저항성에 가까움
		- 예) 얼굴 인식에서 특정 hideen unit이 '코'를 학습했다면, 이 unit을 없앴을 때 코가 없는 얼굴 이미지도 인식할 수 있음 

## 7.13 Adversarial Training

- 정답률이 거의 100프로에 도달한 모델은 주어진 인간 수준의 이해도를 가진다고 얘기하지만, 사실은 아닐수도 있다.
- 이를 테스트 해보기 위해 기존의 데이터에 아주 작은 노이즈를 추가하여 output을 뽑아보면 인간의 대답과는 다소 다른 결과를 내는 경우가 많다.
    - 판다 이미지에 노이즈를 추가한 경우.

        ![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_8.png)

        사람이라면 당연히 판다라고 대답할 이미지. 하지만 뉴럴넷은 긴팔 원숭이라고 대답.

    - Ian Goodfellow에 따르면, 과도한 linearity가 이러한 'adversarial example'을 맞추지 못하는 이유 중 하나라고 한다.
    - 선형 함수는 input이 조금만 변해도 결과가 크게 달라지기 때문에, 'adversarial examples'을 학습시킴으로써  neighbor input에 대해서는 locally constant 한 성격을 가지도록 할 수 있다.
- Adversarial training의 motivation
    - 서로 다른 class의 데이터는 다른 manifold에 존재할 것
    - 어떤 데이터에 small perturbation이 가해진 데이터는 다른 manifold로 넘어갈 수 없다.
- Adversarial example은 또한 준지도 학습에 사용할 수 있다.
    - 라벨이 없는 데이터를 학습된 분류기에 넣어서 output $\hat y$을 만든다.
    - 라벨 $y^{\prime} \ne \hat y$ 을 가지는 adversarial example을 찾는다.
    - 두 데이터가 같은 output을 가지도록 학습시킨다.

## 7.14 Tangent Distance, Tangent Prop, and Manifold Tangent Classifier

"데이터들은 low-dimensional manifold에 존재할 것이다." 라는 가정을 통해 차원의 저주를 극복하고자 하는 세가지 알고리즘을 소개.

- Tangent distance algorithm
    - 두 데이터 $\boldsymbol x_1 , \boldsymbol x_2$ 가 동일한 class에 속하는지를 판별할 때는 두 데이터의 manifold $M_1, M_2$사이의 거리를 이용하는 것이 좋음.(동일한 클래스라면 동일한 manifold에 속할테니)
    - 두 manifolds 에 속하는 모든 데이터 pair $(\boldsymbol x_1 \in M_1, \boldsymbol x_2\in M_2)$에 대해 거리를 구한 후 가장 짧은 거리를 찾는 일은 너무 힘들기 때문에 임의로 정한 하나의 데이터에 대해 tangent plane을 구하여 거리를 구함.

        ![_config.yml]({{ site.baseurl }}/assets/ch7/tangent distance.png)

        ([http://yann.lecun.com/exdb/publis/pdf/simard-00.pdf](http://yann.lecun.com/exdb/publis/pdf/simard-00.pdf))

- Tangent propagation algorithm
    - 기존의 데이터를 아주 조금 변환 시켜도 신경망이 invariant하도록 regularization penalty를 주는 방법.

        $f(\boldsymbol x)$ : output of $\boldsymbol x$,   $\boldsymbol v^{(i)}$: tangent vectors at $\boldsymbol x$

        $$\Omega(f) = \sum_{i}\left(\left(\nabla_{\boldsymbol x}f(\boldsymbol x)\right)^{\top}\boldsymbol v^{(i)}\right)^2$$

        ![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_9.png)

    - Tangent propagation은 변환된 데이터를 통해 학습시킨다는 점에서 data augmentation과 동일하지만, 변환의 정도에서 차이가 난다.
        - tangent propagation은 infinitesimal transformation만 하기 때문에 larger perturbation에 대해서는 regularization이 어렵다.
        - 또한 linear unit을 사용하는 모델은 학습이 어렵다.

- Manifold tangent classifier
    - Auto-encoder를 사용하면 manifold tangent vector를 추정할 수 있다.(14 단원에서 다룰 예정)
    - 즉, Tangent propagation과 동일한 알고리즘이지만 탄젠트 벡터를 사용자가 지정하는 것이 아닌 auto-encoder를 통해 구해진 벡터를 사용.
