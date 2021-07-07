- 딥러닝은 알고리즘을 형식적으로 표현하는 formalism에 기반해 발전해왔는데, structured probabilistic model 역시 이 중 하나임

  - Part II에서 간단히 소개한 바에 더해서, Part III에서 structured probabilistic model은 여러 연구 주제에 필수적으로 포함되는 요소임

- Structured probabilistic model은 그래프를 이용해 확률 분포를 표현함으로써, 랜덤 변수들 사이의 상호 작용을 나타내는 방법임

  - 그래프를 이용하기 때문에 graphical model이라고도 불림

  - 이 단원에서는 다양한 graphical model들을 관통하는 중심 아이디어에 대해 소개할 예정임

  - 또한 16.7에서는 graphical model을 딥러닝에 적용하는 몇 가지 특수한 방법들에 대해 소개할 예정임



## 16.1 The Challenge of Unstructured Modeling

- 딥러닝의 핵심 과제 중 하나는 복잡한 구조를 가진 고차원의 데이터를 이해하는 것임 (ex-자연 이미지, 음성 언어 등)


- 분류 알고리즘은 이러한 고차원의 분포에서 인풋을 받아, 라벨이라는 저차원 데이터로 요약함

  - 이 과정에서 인풋에 담겨있는 많은 양의 정보를 무시하거나 버리게 됨 (ex-사진의 물체를 인식하는 도중 배경은 무시함)


- 한 편, probabilistic model을 이용하면 데이터의 전반적인 구조를 좀 더 자세히 이해함으로써, 단순 분류를 넘어 많은 작업들을 수행할 수 있음

  - Density estimation, denosing, missing value imputation, sampling 등

![_config.yml]({{ site.baseurl }}/assets/ch16/Fig16_1.PNG)


- 수백만 개의 랜덤 변수들에 대해 복잡한 (rich) 분포를 모델링하는 것은 어렵고, 많은 양의 연산이 소모되는 작업임

  - Binary 변수에 대한 간단한 모델링에서도, 32 x 32 RGB 이미지라면 전체 경우의 수가 $2^{3072}$ 나 됨

  - $k$개의 값이 가능할 때, $n$개의 변수로 이루어진 벡터 $x$를 만드는 경우의 수는 $k^n$가 되는데, 이를 모두 파악하는 건 아래의 이유로 적절한 방법이 아님

    - Memory, statistical efficiency, runtime (inference, sampling)


- 하지만 현실에서 대처하는 많은 문제에서는 각각의 변수 사이의 상관 관계가 있기 때문에, 위와 같이 모든 조합을 고려하는 table-based 방법을 취할 필요가 없음

  - 예) 릴레이 계주에서 두번째 이후 주자의 통과 기록은 이전 주자의 기록에 영향을 받을 수 밖에 없음

  - 하지만 3번 주자와 1번 주자의 기록 사이의 영향은 간접적(indirect)인데, 2번 주자의 기록에 큰 영향을 받기 때문임

  - 따라서 계주 기록 모델링에서는 1-2, 2-3 주자 사이의 상호 작용만 고려하면 되고, 1-3 주자 사이의 상호 작용은 고려할 필요 없음


- Structured probabilistic model에서는 랜덤 변수 사이의 직접적인(direct) 상호 작용을 파악함

  - 이를 통해 더 적은 양의 데이터에서 추출된 적은 수의 변수를 가지고도, 더 적은 양의 연산으로 모델을 작동할 수 있음

## 16.2 Using Graphs to Describe Model Structure

- 구조화된 확률 모델은 그래프로 나타낼 수 있다.
  - 노드: 확률변수
  - 엣지: 확률변수 사이의 관계
- 확률 모형을 그래프로 나타내는 여러가지 방법이 있다. 이번 절에서는 그 중 많이 사용되는 방법들을 소개한다.

### 16.2.1 Directed Models

- 그래프 모델은 크게 방향성이 있는 경우와 없는 경우로 나눌 수 있다. 이번 절에서는 방향성이 있는 경우의 모델을 다룬다.
  - directed model, belief network, Bayesian network 등으로 불린다.

![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/16.2.1_1.png)

- Directed model은 변수 사이의 조건부 확률을 화살표로 나타낸다. 식으로 표현하면 다음과 같다.
  - $p(\vec{x}) = \prod_i (x_i \vert Pa_G(x_i))$
  - $Pa_G(x_i)$는 $x_i$의 부모 노드들을 의미한다.
  - 위의 예시 그림을 식으로 표현하면 $p(t_0, p_1, p_2) = p(t_0)p(t_1 \vert t_0)p(t_2 \vert t_1)$

- directed model을 사용하면 더 적은 매개변수를 이용하여 모델을 표현할 수 있다.
  - 예를들어 위 그림에서 $t_0, t_1, t_2$ 모두 100개의 가능한 값 중 하나를 가지는 불연속 확률변수라고 생각해보자. 만약 $p(t_0, t_1, t_2)$를 확률표로 나타내면 총 $999,999=100\times100\times100 - 1$개의 매개변수가 필요하다(전체의 합이 1이므로 하나는 기록하지 않아도 됨). 하지만 directed model을 이용하여 $p(t_0, t_1, t_2) = p(t_0)p(t_1 \vert t_0)p(t_2 \vert t_1)$로 나타낸다면 $99+2\times(100\times100 - 1)$개의 매개변수로 모델을 표현할 수 있다.
  - 즉, 일반적으로 $k$개의 값 중 하나를 가지는 불연속 확률변수 $n$개를 모델링하기 위해 $O(k^n)$개의 매개변수가 필요하다. 하지만 directed model을 사용한다면 $O(K^m)$ ($m$은 하나의 조건부 확률 표현에 등장하는 최대 변수의 개수)개의 매개변수만 이용하여 표현할 수 있다.

- directed model에 모든 종류의 정보를 녹여낼 수 있지는 않다.
  - 예를들어 위 그림에서 Bob은 완주하는데 항상 10초가 걸린다고 하자. 즉 $t_1 = t_0 + 10$이다. 이는 확률 모델의 파라미터 개수를 줄이는 아주 중요한 정보이지만 directed model 자체에는 반영되지 않는다. 이는 확률표에만 반영된다($p(t_1 \vert t_0)$를 표현하는 확률표에 숫자가 하나만 있으면 됨).


### 16.2.2 Undirected Models

- Undirected model은 상호작용의 방향성이 없는 경우에 유용한 모델이다.
  - Undirected models, Markiv random fields, Markov networks 모두 다 같은 말이다.

![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/16.2.2_1.png)

- Undirected model은 변수 사이의 상관관계를 직선으로 표현한다. 식으로는 다음과 같다.

  - $\tilde{p}(\vec{p}) = \prod_{C\in G} \phi(C)$

  - Clique $C$는 그래프 $G$에서 노드들이 모두 서로 연결된 부분집합을 의미.

    (clique 예시 그림)

  - $\phi(C)$는 $C$에 속한 노드들 사이의 상호작용 강도 (affinity)를 의미한다.

    - 이 값은 항상 non-negative이다.
    - 확률값은 아니다.
    - 물리학의 potential에 해당하는 값이다.

- $\phi(C)$를 표현하는 affinity 표는 unnormalized probability distribution이다.

  - 예를들어 $h_y$와 $h_c$를 포함하는 clique의 affinity 표는 다음과 같다.

    |         | $h_y=0$ | $h_y=1$ |
    | ------- | :-----: | :-----: |
    | $h_c=0$ |    2    |    1    |
    | $h_c=1$ |    1    |   10    |


### 16.2.3 The Partition Function

- Undirected model을 표현하는 $\tilde{p}(\vec{x})$는 normalization이 되지 않은 상태이다.
- 이를 확률값으로 변환하기 위해서는 normalize factor가 필요한데 이를 partition function($Z$)이라 부른다.
- 즉, undirected model의 확률 분포는 다음과 같이 쓸 수 있다.
  - $p(\vec{x})=\frac{1}{Z}\tilde{p}(\vec{x})$
  - $Z=\int\tilde{p}(\vec{x})d\vec{x}$

- $Z$는 모든 경우의 수에 대한 합(또는 적분)이다. 일반적으로 계산량이 많아 계산이 불가능하다.
- 딥러닝의 관점에서도 $Z$는 계산이 불가능하다. 18단원에서는 이른 근사하는 방법을 소개한다.

- $Z$가 수학적으로 항상 계산 가능한것은 아니다. Clique potential의 형태에 따라 $Z$가 발산하는 경우도 있다.
  - 예를들어 $\phi(x)=x^2$, $Z=\int x^2 dx$인 경우 $Z$가 발산한다.

- $Z$ 값은 clique potential은 함수꼴, 파라미터, 확률변수의 정의역에 의해 값이 크게 달라진다는 것을 항상 기억해야 한다.
  - 함수꼴에 의해 값이 크게 달라지는 예는 위에서 보였다.
  - 파라미터에 의해서도 값이 크게 달라진다. 예를들어 $\phi(x;\beta)=\exp(-\beta x^2)$인 경우를 생각해보자.
    - $\beta \leq 0$인 경우 $Z$가 발산
    - $\beta > 0$인 경우 $p(x)$는 Gaussian distribution.
  - 확률 변수의 정의역에 의해서도 $Z$와 $p(x)$의 값이 달라진다. $n$차원 확률변수 $\vec{x}$에 대해 $\phi^{(i)}(x_i)=\exp(b_i x_i)$인 경우,
    - $\vec{x} \in \{0, 1\}^n$이면 $p(x_i=1)=\text{sigmoid}(b_i)$.
    - $\vec{x} \in \{[1, 0, ..., 0], [0, 1, ..., 0], ..., [0, 0, ..., 1]\}$이면 $p(\vec{x})=\text{softmax}(\vec{b})$.


### 16.2.4 Energy-Based Models

- Undirected model은 수학적으로 $\tilde{p}(\vec{x}) > 0$라는 가정이 필요하다.
- Energy-based model은 이러한 조건을 만족시키는 쉬운 방법 중 하나이다.
  - $\tilde{p}(\vec{x}) \exp{(-E(\vec{x}))}$
  - 위 식은 Boltamann distribution이기 때문에 Boltzmann machine이라고도 불린다.
  - $E(\vec{x})$ 앞의 $-$는 통계물리학의 컨벤션을 따르기 위해 있는 것이다. 머신러닝 관점에서는 꼭 있을 필요가 없지만(모델 입장에서는 $E(\vec{x})$의 부호를 뒤집어서 학습하면 됨) 보통 $-$를 붙여서 쓴다.

- Energy-based model을 사용하면 학습이 간단해진다.
  - Clique potential을 직접 학습하는 경우 $\tilde{p}(\vec{x}) > 0$를 제약조건으로 가지는 constrained optimization을 해야 한다.
  - 반면 energy-based model을 사용하여 $E(\vec{x})$를 학습하는 경우 unconstrained optimization을 할 수 있다.

- 많은 경우 $p_{\text{model}}$를 계산하지 않아도 된다. 대신 $-\log \tilde{p}(\vec{x})$꼴을 많이 사용한다.
  - 특히 latent variable이 있는 경우 $-\log \tilde{p}(\vec{x})$를 free energy라고 부른다.
  - $F(\vec{x}) = -\log \tilde{p}(\vec{x}) = -\log \sum_{\vec{h}}\exp(-E(\vec{x}, \vec{h}))$

### 16.2.5 Seperation and D-Seperation

Graphical 모델을 사용할 때 변수들 사이의 조건부 독립성을 판단하는 두 가지 방법이 있다. 조건부 독립을 짧게 정의하면 다음과 같다.

"변수들의 부분집합 $\mathbb S$가 주어졌을 때 서로 다른 두 부분집합 $\mathbb A$와 $\mathbb B$가 독립이면 $\mathbb A$와 $\mathbb B$는 $\mathbb S$에 대해 조건부 독립이다."

1. Seperation

   - Undirected 그래프에서 사용.

   - 관측된 변수를 지나는 path는 inactive, 관측되지 않은 변수를 지나는 path는 active라고 하며, 두 변수 $a$와 $b$ 사이에 active path가 없을 때 seperated 되었다고 한다.

     ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_6.png)


2. D-seperation

   - Directed 그래프에서 사용.

   - 'D' 는 dependence를 의미한다.

   - Seperation과 비슷하게 active path가 없을 때 d-seperated 되었다고 하지만 그래프에 방향성이 있기 때문에 조금 복잡하다.

   - 방향성이 일정한 그래프에서는 undirected의 경우와 마찬가지로 변수 $s$가 관측되면 $a$와 $b$는 분리된다.

     ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_8(a).png)

   - 두 변수 $a,b$ 모두 $s$로 부터 파생된 경우. $s$가 관측되면 $a$와 $b$는 분리되지만, 관측되지 않은 경우에는 $a$와 $b$가 의존할 수 있다.

     ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_8(b).png)

   - $a$와 $b$ 모두 $s$의 부모인 경우. $s$가 관측되면 $a$와 $b$는 의존한다. 예를 들어, $a$를 '동료가 휴가를 감', $b$를 '동료가 병가를 냄' 그리고 $s$를 '동료가 결근을 함'이라 해보자. $s$가 관측되면 결근한 이유가 휴가를 갔거나 병가를 냄 둘 중 하나이기 때문에 $a$와 $b$는 의존한다.

     ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_8(c).png)

   - $(c)$와 같은 구조에서 $s$의 자식 $c$ : '동료로부터 결과 보고서를 받음'가 있을 때, 동료로부터 결과 보고서를 받지 못했다면 동료가 결근했을 가능성이 올라가기 때문에 $(c)$와 같은 이유로 $a$와 $b$는 의존한다.

     ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_8(d).png)

   물론 그래프 형태만 보고 변수들의 모든 의존관계를 알 수는 없다. 대표적인 경우가 context-specific independences이다. 예를 들어 세 이진 변수 a,b 그리고 c가 있을 때, a가 0이면 b와 c는 독립, a가 1이면 b=c라고 하자. a=1 일 때를 그래프로 나타내면 b와 c는 연결되어 있어야한다. 하지만 이럴 경우 a=0인 경우는 나타낼 수 없다. 

### 16.2.6 Converting between Undirected and Directed Graphs

Directed 모델과 undirected 모델 둘 다 장단점이 있지만 자신이 수행하고자 하는 작업에 맞게끔 정하는 것이 중요하다. 두 모델 중 어떤 것을 사용할 지는 아래의 두 접근 방식으로 결정할 수 있다.

- 확률분포에서 가장 많은 독립성을 표현할 수 있도록
- Edge가 가장 적어지도록

Complete graph는 어떤 확률 분포든 표현 가능하지만 변수들 사이의 독립성을 표현할 수 없기 때문에 좋은 선택은 아니다.

또한 두 모델은 서로 전환이 가능하다. 

- Directed 모델에서 undirected 모델로 전환할 때는 모든 edge의 방향성을 없애는 것 뿐만 아니라 추가적인 edge를 연결 해야한다.

  ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_11.png)

  왼쪽 그래프처럼 직선관계의 그래프는 방향성만 없애주면 되지만, 중간과 오른쪽 그래프처럼 두 변수(a, b)가 모두 하나의 변수(c)의 부모인 경우에는 부모들 사이에도 edge를 연결해 주어야한다.(이런 구조를 immortality라 부른다.) 따라서 이런 경우에는 a와 b사이의 독립성을 잃어버리게 된다. 이렇게 만들어진 undirected 그래프를 moralized 그래프라고 한다.

- Undirected 모델에서 directed 모델로 전환할 때는 loop를 조심해야 한다. 길이 4 이상의 loop는 chord를 추가하여 삼각형화(?) 해주어야 한다고 한다. 그 후에 방향성을 주되 directed cycle이 생기지 않도록 만들어 주어야 한다.

  ![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_12.png)

### 16.2.7 Factor Graphs

요소 그래프는 undirected 그래프에서 하나의 clique를 정규화 되지 않은 확률분포 $\phi$로 나타내는 방법이다. 

![_config.yml](/Users/kibum_onepredict/dev/deeeperlearning.github.io/_posts/{{ site.baseurl }}/assets/ch16/Fig16_13.png)

위의 그림처럼 확률변수 a,b 그리고 c가 clique를 이루고 있다고 하면 다음과 같이 요소 그래프로 나타낼 수 있다.

- (Center) (a,b,c)를 인자로 받는 확률분포 $f_1$이라는 하나의 요소로 나타낸 것이다. 요소는 사각형 노드로 나타내며 인자로 받는 확률변수는 모두 $f_1$에 연결되어 있어야 한다.
- (Right) 두 개의 확률변수를 인자로 받는 확률 분포를 요소로 사용한 경우. 이 경우 또한 각 요소는 인자로 받는 확률변수와 연결 되어 있어야 한다.



## 16.3 Sampling from Graphical Models

- Directed graphical model의 장점 중 하나는 ancestral sampling을 이용해 간단하게 샘플을 만들을 만들어낼 수 있다는 것임

  - Ancestral sampling: 변수들의 순서가 정해지면 $P(x_1)$을 샘플링 한 뒤, 귀납적으로 $P(x_{n} \mid x_{n-1})$을 샘플링해 최종 아웃풋을 만들어냄

  - 어떤 그래프에서는 하나 이상의 순서가 가능한데, ancestral sampling는 모두에 적용될 수 있음

  - 계산이 간단하기 때문에 빠르고 편리하게 샘플링을 수행할 수 있음
- Ancestral sampling의 한 가지 단점은 directed graphical model에만 적용될 수 있다는 것임

  - Undirected model을 directed model로 바꿔서 사용할수도 있지만, 일반적으로는 매우 복잡한 그래프가 되어 다루기 힘든 문제가 됨
  - Directed model로 변환하지 않는다면, cyclical dependency를 해결해야 함
  
    - 모든 변수들이 모든 다른 변수들과 상호작용하여, 샘플링을 위한 분명한 시작점이 없게됨
  - 이와 같이 undirected graph를 이용한 샘플링은 일반적으로 소모적인 작업인데, Chapter 17에서 더 자세히 다룰 예정임



## 16.4 Advantages of Structured Modeling

전술하였듯, probabilistic model을 사용할 때 얻을 수 있는 가장 큰 장점은 학습 및 추론 비용의 절약이다. directed model의 경우 sampling 속도도 빨라진다. graphical model은 edge들을 제거하는 형태로 정보를 부호화한다. 두 node가 edge로 연결되지 않는다는 것은 둘 사이의 직접 상호작용을 모형화할 필요가 없다는 것이다.

또 다른 장점으로, 학습으로 얻은 지식의 표현(weight)과 기존 지식에 기초한 추론으로 얻은 지식의 표현(edge-cutting)을 명시적으로 분리할 수 있다는 것이다.

## 16.5 Learning about Dependencies

좋은 generative model은 visible(관측 가능한) variable v에 관한 분포를 정확하게 포착해야 한다. 일반적으로 그런 v의 다양한 성분들 사이의 의존성이 존재하는데, 이 의존성을 모델에 포함시키는 가장 흔한 방식이 hidden latent variable h를 도입하는 것이다. h를 도입한 모형은 $v_i$와 $v_j$ 사이의 간접적인 종속 관계를 $v_i$→h와 h→$v_j$와 같은 직접적인 종속 관계로 분해해 포착할 수 있다.

직접 연결된 visible variable 사이의 의존성을 model에 포함시키려는 경우, 모든 변수를 all-to-all로 연결하는 것은 너무 코스트가 크므로 서로 밀접하게 연관된 edge들만 연결되도록 해야한다. 이 때 선험적인 지식이 필요할 것이다. 이 문제만 다루는 structure learning이라는 ml 분야가 있다.

## 16.6  Inference, Approximate Inference

Probabilistic model의 주 용도는 variable 사이의 관계를 알아내는 것이다.

latent variable model에서는 observed variable v를 서술하는 feature  $E[h\mid v]$를 추출해야 할 때가 있다. 일반적으로 이런 경우 MLE를 계산하게 되며, 결과적으로 $p(h\mid v)$를 계산해야 한다. 다시 말해 어떤 변수들의 값이 주어질 때 다른 변수들의 값이나 그에 대한 확률분포를 예측하는 추론 문제라고 할 수 있다. 허나, 심층 모형에서 이런 추론 문제는 (analytic하게?) 처리 불가능이다. 그리하여 approximate inference(근사추론) 기법들이 나왔다. 이런 추론을 variational inference(변분추론)이라고도 부른다.

variation inference에서는 진짜 분포 $p(h\mid v)$와 새로 도입하는 분포 $q(h\mid v)$가 최대한 가까워지도록  $q(h\mid v)$를 학습한다.

## 16.7 The Deep Learning Approach to Structured Probabilistic Models

DNN에 쓰이는 probabilistic model에서는 주어진 DNN의 깊이를 계산 그래프가 아니라 그래프 모형을 기준으로 정의한다. latent variable $h_i$와 어떤 visible variable 사이의 최단 경로의 길이가 j라고 할 때 j를 $h_i$의 깊이로 간주할 수 있다.

전통적인 그래프 모형의 경우 변수들이 대부분 가끔은 관측되는 변수들로 구성되어 있고, 각각의 변수들에 의미를 부여하나, DNN에서는 latent variable들이 거의 항상 존재하며 구체적인 의미를 부여하지 않는 경우가 많다. 예를 들어, 전통적인 그래프 모형에서는 latent variable의 구체적인 의미-지능, 환자 증상 등-을 미리 연구자가 정해둔다고 생각하면 된다.

### 16.7.1 Example: Restricted Boltzmann Machine

graphical model을 깊은 학습에 사용하는 예로 RBM을 들 수 있다. RBM은 generative model로 데이터들의 latent factor들을 확률적인 방법으로 얻어낼 수 있는 모델이다.

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled.png)

RBM은 binary visible unit과 binary hidden unit의 두 층으로 구성된 energy based Model이다. 이 모형의 energy function은 아래와 같다.

$$E(v, h) = - b^Tv - c^Th - v^T Wh$$

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled1.png)

b, c, W는 제약이 없는 학습 가능한 real number parameter들이다. h layer와 v layer의 관계는 W로 서술된다.

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled2.png)

hidden layer h 사이, 그리고 visible layer v 사이에는 직접적인 상관관계가 존재하지 않는다고 가정한다.  이런 제한으로부터 아래와 같은 성질이 나온다.

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled3.png)

이는 아래와 같이 계산되고,

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled4.png)

따라서 $P(h_i =1 \mid v)$는 아래와 같이 계산될 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled5.png)

에너지함수는 매개변수들의 선형 조합이므로 미분을 구하기 쉽다. 예를 들어

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled6.png)

모형을 훈련하면 v에 대한 expression h가 유도된다.

![_config.yml]({{ site.baseurl }}/assets/ch16/Untitled7.png)

RBM은 graph 모형에 대한 DNN의 접근방식을 잘 보여준다.