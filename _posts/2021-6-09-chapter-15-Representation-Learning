- 이 단원에서는 representation을 학습한다는 것의 의미와 이 개념이 깊은 신경망 구조를 디자인하는데 왜 유용한지에 대해 다룰 예정임

   - 학습 알고리즘이 통계적인 정보를 어떻게 다른 과제 사이에서 공유하는지

   - 비지도 학습에서 얻어진 정보를 어떻게 지도 학습 과제를 위해 사용하는지



- Representation을 공유하는 것은 여러 도메인을 한 번에 다루거나, 학습된 정보를 훈련용 데이터가 부족한 과제에 적용시킬 때 도움이 됨

   - 이는 distributed representation(Hinton et al., 1986)에서부터 시작된 논의로, 15.4에서 다룰 예정임



- 정보를 어떻게 표현하는지에 따라 기계학습 과제는 매우 쉬워질수도, 매우 어려워질수도 있음

   - 예1) 210을 6으로 나눌 때 long 타입을 적용한다면?

   - 예2) 숫자를 sorted list의 알맞은 위치에 집어넣을 때,
      
      - Linked list라면 $O(n)$의 연산이 필요함
      
      - Red-black tree라면 $O(log n)$의 연산이 필요함



- 그렇다면, 무엇이 좋은 representation일까? $\rightarrow$ 과제를 쉽게 만드는 representation

   - 따라서 과제에 따라 알맞은 representation도 변하게 됨


- 예를 들어, 지도 학습으로 훈련된 앞먹임 신경망을 representation 학습이라 할 수 있음

   - 마지막 층은 softmax regression 분류기와 같은 선형 분류기이고, 나머지는 이 분류기에 제공하기 위한 representation을 학습함

   - 지도 학습 과정은 자연스럽게 모든 은닉 계층에 representation을 발생시킴 (상위의 은닉 계층에 더욱 가공된 고급 정보가 저장됨)
   
   - 따라서 인풋에서는 선형적으로 separable 하지 않던 클래스들이, 마지막 은닉 계층에서는 separable 해질 수 있음



- 대부분의 representation 학습는 최대한 많은 정보의 보존과 좋은 성질을 획득하는 작업 사이의 tradeoff를 맞게됨
   
   - 비지도 혹은 semi-지도 학습이 가능하도록 하는 점 또한 representation 학습의 흥미로운 부분임

   - 일반적으로 많은 양의 unlabeled, 적은 양의 labeled 학습 데이터를 이용하게 됨

   - 적은 양의 labeled 데이터에 지도 학습을 적용하면 심각한 overfitting이 발생하기도 함

   - 이 때, semi-지도 학습은 unlabel 데이터에서부터도 학습을 함으로서 overfitting 문제를 해결할 수도 있음

   - 요약하면, unlabeled 데이터에서부터 좋은 representation을 학습하고, 이를 지도 학습 과제에 적용하게 됨



- 인간이나 동물은 매우 적은 양의 labeled 데이터에서도 학습이 가능한데, 어떻게 가능한지는 알려진 바가 없음

   - 뇌가 비지도, 혹은 semi-지도 학습을 이용하여 레버리지를 작동시킬 수도 있음

   - Unlabeled 데이터로부터 좋은 representation을 학습하는 레버리지를 작동시키는 여러 방법에 대해서 다룰 예정임



## 15.1 Greedy Layer-Wise Unsupervised Pretraining

- Greedy layer-wise unsupervised pretraining : 합성곱이나 recurrence와 같은 특수한 구조가 없이도 지도 학습을 위한 신경망을 학습시키는 방법

   - 한 과제에서 학습한 representation(인풋 분포를 파악하는 비지도 학습)을 다른 과제에 이용(같은 인풋 도메인에 대한 지도 학습)함



![_config.yml]({{ site.baseurl }}/assets/ch15/Greedy.PNG)



- "Greedy ... pretraining"은 한 계층에 대한 학습 알고리즘으로 작동함 (RBM, single-layer autoencoder 등)

   - 각 계층은 비지도 학습을 이용해 pretraining되고, 이전 계층의 아웃풋으로부터 데이터의 새로운 (그리고 아마도 간단한) representation을 만들어냄

   - 지도 학습 과제를 위한 신경망에 접목시키는 과정이 어렵다고 여겨져왔으나, 2000년대에 접어들어 딥러닝 르네상스를 맞으며 좋은 initialization 조건을 만드는 방법으로 관심받고 있음



- "Greedy(탐욕스러운)"이라고 불리는 이유는 솔루션의 모든 조각들을 독립적으로 최적화하기 때문임

   - "layer-wise"는 이 각각의 독립된 조각들이 신경망의 개별 계층들이기 때문임

   - 다시 말해 한 번에 한 계층씩 작동하며, 이 때 모든 이전 계층들은 고정되어 있음

   - 각 계층이 비지도 학습 알고리즘으로 학습되어 "비지도"가 붙지만, 동시에 labeled 데이터를 이용한 fine-tuning을 실행하기 전이므로 pretraining이 붙음



### 15.1.1 When and Why Does Unsupervised Pretraining Work?

Greedy layer-wise 비지도 사전학습은 분류 작업에서 큰 성능향상을 보여주지만 다른 작업에서는 오히려 해로운 효과를 야기할 수 있기 때문에, '언제' 그리고 '왜' 이 방법이 성능향상을 가져오는지 알아야 한다.

비지도 사전학습은 두가지 아이디어를 내포하고 있다.

- 심층학습 네트워크에 사용될 파라미터의 초기값에 따라 정규화 효과를 기대할 수 있다.
    - 아직 파라미터 초기값 설정에 대한 이해가 부족하고 deep neural network가 완벽한 minimum에 데려다 주지 않는다는 것을 안다.
    - 하지만, 사전학습이 우리가 도달하지 못하는 영역에 데려다 줄 수 있다는 가능성은 있다.

        ![_config.yml]({{ site.baseurl }}/assets/ch15/Fig15_1.png)

        위 그림은 Erhan et al. (2010)의 실험 결과이다. 서로다른 초기값(각각의 점)에서 시작한 모델들의 시간에 따른 학습 궤적을 function space에 나타낸 것이다. 색깔은 시간을 의미한다. 모든 점들은 그림의 중앙에서 시작하여 바깥쪽으로 뻗어나가는데 사전학습을 한 경우에는 궤적의 분산이 매우 작다. 즉, 뉴럴넷의 stochasticity를 줄일 수 있다.

- 입력 데이터의 분포를 학습하는 것은 입력과 출력 사이의 mapping을 학습하는데 도움이 된다.
    - 비지도 학습에서 유용한 feature는 지도 학습에서도 유용할 것이다. 예를 들어, 자동차와 오토바이를 분류하기 위해선 바퀴의 특징과 바퀴의 개수를 파악할 수 있어야하는데, 비지도 사전학습이 바퀴에 대한 좀 더 쉬운 representation을 찾아준다면 분류가 쉬워질 것이다.
    - 하지만, 아직 수학적으로 증명된 것이 아니고 어떤 모델을 사용하는가에 대한 의존성도 너무 크다. 예를 들어, 지도학습에 선형 분류기를 사용한다면 사전학습된 feature는 학습할 클래스들을 선형적으로 분류할 수 있어야 한다.

또한 비지도 사전학습을 아래와 같이 두가지 관점으로 생각해 볼 수 있다.

- Representation 학습으로 바라보는 관점
    - 초기의 representation이 좋지 않을수록 큰 효과를 기대할 수 있다.  One-hot 벡터로 단어를 embedding 시킨다면 인접한 두 단어 사이의 거리는 항상 일정하지만, 학습된 word embedding은 자연스럽게 두 단어 사이의 유사성을 학습하게 된다. 이런 경우, 비지도 사전학습은 좋은 효과를 기대할 수 있다.
- Regularizer로 바라보는 관점
    - 비지도 사전학습을 통해 추가된 정보는 모두 라벨이 없는 데이터로부터 왔기 때문에 라벨이 없는 데이터가 많을수록 큰 효과를 기대할 수 있다.
    - 또한, Weight decay 같은 정규화와는 다르게 비지도 학습은 feature functions을 학습하기 때문에, 학습해야 할 함수가 굉장히 복잡할 때 이로울 수 있다.

비지도 사전학습을 통해 위와 같은 이로운 효과를 기대할 수 있지만, 학습을 두 스테이지로 나눔으로써 생기는 단점도 있다.

- 정규화 효과에 관여하는 하이퍼파라미터가 너무 많고, 학습이 끝난 후에야 측정이 가능하다.
- 비지도 사전학습에 사용되는 하이퍼파라미터를 튜닝하기 위해서는 지도학습까지 끝내야 하므로 시간이 오래걸린다.

사실 요즘에는 dropout, batch normalization 등등 좋은 기술들이 많고 성능도 더 좋아서 비지도 사전학습은 NLP쪽 빼고는 거의 사용하지 않는다고 한다...


## 15.2 Transfer Learning and Domain Adaptation

Transfer Learning (TL) 과 Domain Adaptation(DA)은 하나의 설정(예를들어 분포 $P_1$)에서 학습한 것을 다른 설정(분포 $P_2$)의 일반화를 개선하기 위해 활용하는 것을 말한다.
이번 절에서는 TL과 DA에 대해 구체적인 구현 방법을 소개하기보다는 추상적인 개념들을 전달한 후 예시를 드는 방식으로 설명할 예정이다.

#### Transfer Learning

- $P_2$의 학습을 위해 학습해야 하는 지식이 $P_1$의 지식과 관련이 있는 경우 사용할 수 있다.
- 특히 $P_2$의 데이터셋이 작고 $P_1$의 데이터셋은 굉장히 크다면 $P_2$에 대한 모델의 일반화 성능에 크게 도움을 줄 수도 있다.
- TL은 이미지 관련 분야에서 활발하게 사용된다. 두 분포에서 풀어야 하는 문제가 다르더라도 (e.g. $P_1$에서는 사진을 개와 고양이로 구분, $P_2$에서는 사진을 호랑이와 사자로 구분) 윤곽선이나 기하도형, 기하학적 변화나 조면 변화의 효과 같은 저수준 개념들을 공유하기 때문에 일반화 개선이 가능하다.

![_config.yml]({{ site.baseurl }}/assets/ch15/15_2_1.png)

- 위 그림처럼 서로 다른 과제들이 모델의 입력 부분이 아닌 출력 부분을 공유해야 할 때도 있다. 이럴 경우 모델의 입력 부분은 여러 갈래로 분리하고 출력만 공유하는 형태도 가능하다.
    - 예를들어 여러 언어의 음성을 인식하여 문장으로 바꾸는 모델이 해당될 수 있다. 일단 음성을 latent space로 인코딩 한 후에는 동일한 매핑을 통해 문장으로 바꿀 수 있지만, 입력에서 latent space로 매핑하는 방법은 언어마다 달라져야 한다.
- TL의 두 극단적인 예로는 one-shot learning과 zero-shot learning이 있다.
    - One-shot learning: label이 있는 데이터를 딱 한 샘플만 사용하는 경우.
        - 예시로 보안을 위해 특정 인물의 얼굴만 인식해야 하는 경우가 있다. 이 경우 첫 단계에서 많은 사람 얼굴 사진으로 모델을 학습한 후 TL 단계에서 특정인의 사진 딱 한장("특정인"이라는 label이 있는 샘플인 셈)만 가지고 모델을 학습할 수 있다.
        - 이런게 가능한 이유는 첫 단계에서 데이터의 배경에 깔린 클래스들을 깔끔하게 나누었기 때문이다. 즉, TL 단계에서 label이 있는 샘플이 하나만 주어지더라도 해당 label이 존재할 수 있는 영억을 모델이 추론할 수 있다.
    - Zero-shot learning: label이 없는 데이터만 사용하는 경우.
        - 예시로 대량의 텍스트만 읽은 후 이미지에서 물체를 인식하는 문제가 있다. 만약 모델이 읽은 텍스트 중 고양이를 충분히 잘 서술한 텍스트가 있다면 (e.g. 다리가 네 개이고 귀가 뾰족하고...) 주어진 이미지에 고양이가 있는지 없는지 판단할 수 있다.
        - 비슷한 현상이 기계 번역에서도 나타난다. 언어 X의 문장을 Y의 문장으로 번역하는 모델이 있다고 하자. 언어 X의 단어 A를 Y의 단어 B로 번역하라는 데이터가 없더라도 모델은 이를 추론할 수 있다.
        - 이러한 학습은 아래 그림처럼 한 modality에서의 변수 $x$를 $h_x$로 매핑하는 방법, 또 다른 modality에서의 변수 $y$를 $h_y$로 매핑하는 방법, $h_x$와 $h_y$의 관계를 모두 배우는 셈이다.

![_config.yml]({{ site.baseurl }}/assets/ch15/15_2_2.png)

#### Domain Adaptation

- DA는 한 설정의 분포를 이용해 다른 설정의 일반화를 개선한다는 점은 같지만, 입력의 분포가 약간 달라진다는 면에서 다른다.
- 예를들어 MNIST 데이터로 숫자 인식을 학습한 모델은(윗줄) MNIST의 배경을 무작위 사진으로 바꾼 입력(아랫줄; MNIST-M)에 대해 숫자를 거의 인식하지 못한다.

![_config.yml]({{ site.baseurl }}/assets/ch15/15_2_3.png)

- DA의 목표는 입력의 분포가 조금 달라지더라도 latent space로 매핑되었을 때 분포가 같아지도록 만드는 것이다.
    - 위의 MNIST에는 label이 있지만 MNIST-M에는 label이 없는 경우 MNIST-M의 숫자를 인식해야 하는 문제를 생각해보자. 학습하려는 모델이 크게 입력을 latent vector로 매핑하는 함수 $h = f(x)$와 latent vector를 label로 매핑하는 분류기 $y = g(h)$로 이루어져 있다고 하자. 이 경우 $f(x)$가 MNIST와 MNIST-M을 같은 latent 분포로 매핑할 수 있다면, 분류기는 MNIST로 학습한 것을 그대로 사용할 수 있다.
    - 이렇게 latent space의 분포를 일치시키는 방법은 여러가지가 있지만 크게 두 가지가 있다.
        - Latent space에서 maximum mean discrepancy(MMD)의 차이를 줄이는 방법: MMD는 두 분포의 거리를 재는 방법 중 하나이다. MNIST를 인코딩 한 경우와 MNIST-M을 인코딩 한 경우의 MMD를 줄이도록 학습하면 latent space에서 분포를 일치시킬 수 있다.
        - GAN을 이용하는 방법: discriminator를 하나 두어 MNIST를 인코딩 한 경우의 latent vector와 MNIST-M을 인코딩 한 경우의 lantet vector를 구분하고 하고, 입력을 latent vector로 인코딩 하는 모듈은 discriminator가 이 둘을 구분하지 못하게 한다면 분포를 일치시킬 수 있다. MMD 이후에 많이 사용되는 방법이다.



## 15.3 Semi-supervised Disentagling of Causal Factors

representation learning에서 중요한 건 "어떤 representation이 좋은 건가?"이다. 서로 다른 feature들이 관측된 자료의 서로 다른 원인들에 대응되면서, 서로 풀어해쳐지는(disentangle) 효과를 내는 표현 방식이 이상적일 것이다. representation learning의 다른 접근으로는, 모델링하기 쉬운 representation을 사용하는 방법이 있다. 예를 들어 성분들이 sparse하거나 서로 독립적인 경우가 있겠다. causal factor들이 숨겨져있어 이를 모형화하기 어려운 경우도 많다. pretraining이 도움이 되는 경우와 그렇지 않은 경우의 예시는 아래와 같다. 

- pretraining이 $p(y\mid x)$에 도움이 되지 않는 경우: $p(x)$가 uniform distribution이고 $E[y\mid x]$를 배워야 하는 경우.
- pretraining이 성공적으로 적용될 수 있는 경우: 아래 그림과 같이 $p(x)$가 일종의 mixture of gaussian이라고 해보자. 이 때 이 mixture 분포가 y의 각 값당 하나의 성분 분포를 혼합한 것이라면, $p(x)$를 적절히 모형화하는 경우 y를 찾기 쉬워져 $p(y\mid x)$가 잘 학습된다.

![1]({{ site.baseurl }}/assets/ch15/Untitled1.png)

다시 말해 $y$가 $x$의 원인중 하나와 밀접하게 연관된다면, $p(x)$와 $p(y\mid x)$는 강한 연관관계를 가질 것이다. 이런 경우 pretraining을 동반하는 semi-supervised learning이 유용하게 먹힐 것이다. 무엇을 부호화할 것인지 결정하는 문제는 semi-supervised learning의 주요 연구 과제중 하나다. 현재 연구되는 주된 전략들은 아래와 같다.

- supervised learning signal과 unsupervised signal을 동시에 사용해 모형이 variation의 가장 의미있는 factor을 포착하도록 하는 것
- 더 큰 unsupervised network structure 사용

unsupervised learning의 다른 전략중 하나는 가장 두드러진 underlying cause의 정의를 수정하는 것이다. 즉 고정된 판정 기준을 사용하는 것이 아니라(MSE같은) 변동하는 기준을 사용하는 것이다. 대표적인 예로 Generative adversarial network가 있다. 한 generative model과 "detective" classifier로 구성되어 있는데, generative model은 classifier가 가짜임을 드러내지 못하도록 훈련되고, classifier은 generative로부터 나오는 input이 real input인지 artificial인지 구분하는 모델을 학습한다.

![2]({{ site.baseurl }}/assets/ch15/Untitled2.png)

![3]({{ site.baseurl }}/assets/ch15/Untitled3.png)

Underlying causal factors를 학습하는 전략의 장점은, $x$가 output이고 $y$가 cause라면, $p(x\mid y)$에 대한 모형이 $P(y)$의 변화에 robust하다는 것이다. 다시 말해 underlying causal factor에 대한 marginal distribution은 변하지만, Casual mechanism은 변하지 않는다.



## 15.4 Distributed Representation

분산 표현은 $n$개의 특징을 $k$개의 값으로 나타냄으로써 $k^n$개의 개념을 설명할 수 있다. 많은 딥러닝 모델들이 "히든 유닛이 주어진 데이터를 설명하는 근본적인 요소를 학습할 수 있다" 라는 가정에 영감을 받았기 때문에, 분산 표현을 사용하는 것은 매우 자연스러운 접근이다. 아래 그림을 보면 비분산 또는 희소 표현 과 분산 표현의 차이를 알 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch15/Fig_dist_rep.png)

([https://towardsdatascience.com/distributed-vector-representation-simplified-55bd2965333e](https://towardsdatascience.com/distributed-vector-representation-simplified-55bd2965333e))

비분산 표현은 일반적으로 class를 분류하지만 분산 표현은 각각의 특징에 값을 부여하여 representation space를 나눈다. 즉, 분산표현은 서로 다른 개념(개, 고양이 ...) 사이의 공유되는 특징을 학습하기 때문에 일반화 관점에서 큰 이점을 얻는다. 예를 들어, 분산 표현이 '모피를 가졌는가' 와 '다리의 개수'라는 특징을 포함한다면, 개와 고양이는 두 특징에 대해 같은 값을 가질 것이다. 또한, 단어를 표현할 때 분산 표현이 one-hot 표현 보다 풍부한 similarity space를 제공하기 때문에, 분산 표현을 사용한 언어 모델이 다른 모델보다 일반화가 잘 된다.

비분산 표현은 학습할 함수 $f$에 대해 '$u\approx v$라면 $f(u)\approx f(v)$일 것이다'라는 가정을 기본으로 한다. 하지만, 

데이터의 차원이 증가하고 조금의 변화($x \rightarrow x+\epsilon$)에 굉장히 민감하다면, 서로 다른 카테고리를 분류하기 위해 굉장히 많은 파라미터가 필요하다. 하지만 분산 표현을 사용하게 되면 이러한 복잡한 구조의 함수도 적은 양의 파라미터로 표현이 가능하다.

![_config.yml]({{ site.baseurl }}/assets/ch15/Fig15_7.png)

위 그림은 분산 표현을 사용하여 representation을 나눈 그림이다($n=3, d=2$). 분산 표현을 사용하면$O(nd)$의 파라미터로 $O(n^d)$ 만큼의 서로 다른 영역을 표현할 수 있다(Zaslavsky(1975), Pascanu et al.(2014b)). 따라서 피팅할 파라미터의 양이 줄고, 즉, 일반화 시키기 위한 training example의 수가 줄게 된다.

Radford et al. (2015)의 실험 결과를 보면 컨볼루션 네트워크에서는 이러한 분산 표현이 (항상은 아니지만) 굉장히 직관적이고 해석 가능한 결과를 준다는 것을 알 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch15/Fig15_9.png)

위 그림을 보면 Generative 모델을 통해 학습된 분산 표현이 성별과 안경의 유무를 구분한다는 것을 알 수 있다. 안경을 낀 남성 이미지의 representation 벡터에서 남성 이미지의 벡터를 빼고 여성 이미지의 벡터를 더한 새로운 벡터는 안경을 낀 여성의 이미지를 만들어 낸다. 이러한 일반화된 특징은 학습하지 않은 이미지에 대해서도 적용 될 수 있다.


## 15.5 Exponential Gains from Depth

이 책의 앞에서 신경망이 깊어질수록 얻는 이득에 대해 소개했다. 요약하면 깊은 신경망은 아래와 같은 통계적인 이득을 얻을 수 있다.

- 다층 퍼셉트론의 경우 보편적 근사기이다.
- 일부 함수의 경우 깊은 신경망을 이용하면 얕은 신경망에 더 작은 신경망으로도 같은 함수를 표현할 수 있다.
- 같은 함수를 표현하기 위한 깊은 신경망의 크기는 얕은 신경망에 비해 지수적으로 작다.

이러한 이득은 분산 은닉 표현을 사용하는 다른 종류의 신경망에도 일반적으로 적용된다.

- 15.4 절의 생성 모형
    - 15.4에서는 얼굴 이미지에서 바탕에 깔린 요소들(안경 착용 여부, 성별 등)을 학습한 생성 모형의 예를 보았다.
    - 이러한 인자들은 주로 고차원의 feature들이다.
    - 일반적으로 이러한 고차원 feature들은 모델의 입력과 복잡한 비선형 관계를 이룰 가능성이 크다.
    - 따라서 이러한 고차원 feature들을 표현하기 위해서는 깊은 싱경망을 이용해 여러 인자들을 비선형적으로 조합해야 한다.
- 다수의 비선형성들과 재사용된 feature들을 hiararchical하게 조합하면 통계적 효율성이 지수적으로 증가한다.
    - 물론 (보편적 근사기이미으로) 하나의 은닉층만 가지고도 이를 표현할 수 있지만, 필요한 은닉 단위의 개수가 굉장히 클 수 있다.
- 합성곱 신경망에 대해서 깊은 모델이 가지는 지수적인 이득을 이론적으로 보인 연구도 있다.



## 15.6 Providing Clues to Discover Underlying Causes

- 단원을 마무리하며, 근복적인 질문을 짚어보자: 무엇이 좋은 representation인가?

  - 15.3 복습: 데이터 다양성의 원천이 되는 요소를 파악하는 representation이 이상적임 (특히 주어진 과제와 관련이 있는)

  - Representation 학습을 이용하는 대부분의 전략들은 이러한 요소들을 찾을 수 있도록 단서를 제공하고자 함

  - 이러한 단서들은 특정 요소들을 다른 요소들과 구분짓도록 도와줌



- 지도 학습에서는 굉장히 강력한 단서가 주어짐: label $y$와 적어도 한 요소는 서로 다른 인풋 $x$

  - 일반적으로 많은 양의 미표시 데이터를 사용하기 위해서, representation 학습은 직접적이지 않더라도 이러한 요소들에 담긴 힌트들을 활용함

  - 물론 완벽한 정규화 전략이란건 없지만, 다양한 과제에 적용할 수 있는 꽤 범용성 있는(fairly generic) 정규화 전략을 찾으려는 시도임



- 이러한 측면에서, 기반이 되는 요소를 파악할 수 있도록 다양하게 활용되고 있는 정규화 전략들을 간략하게 소개하며 단원을 마무리함

  - Smoothness
  
    - $f(x+\epsilon d) \approx f(x)$를 가정하여 학습 데이터 근처의 인픗 공간까지로 일반화할 수 있도록 함

    - 많은 기계 학습 알고리즘에 적용되지만, 차원의 저주를 해결할 수 는 없음
    
    
  - Linearity
    
    - 어떤 변수 사이의 관계가 선형이라고 가정함으로써, 관찰된 데이터에서 매우 먼 지점까지도 예상 할 수 있음
    
    - Smoothness 가정을 사용하지 않는 많은 기계 학습 알고리즘들이 linearity를 가정함
    
    - 그러나 비상식적인 결과가 나올수도 있으며, 가중치가 큰 경우 함수가 smooth 해지지 않을 수 있음

    
  - Multiple explanatory factors
  
    - 데이터가 여러 요소에 의해 정해진다면, 각각을 파악함으로써 문제를 해결할 수 있음

    - 예) $p(x)$를 예상하기 위해, $p(y \mid x)$를 파악하기
    
    
  - Causal factors
  
    - 관측된 데이터 $x$의 원인이 되는 representation $h$를 파악함으로써 문제를 해결할 수 있음

    
  - Depth (or a hierarchical organization of explanatory factors)
    
    - 높은 수준의 개념을 얻어내기까지의 여러 앞먹임 단계를 파악하여 문제를 해결할 수 있음
     
    
  - Shared factors across tasks
   
    - 여러 문제를 해결해야 할 때, representation을 공유함으로써 각 문제의 통계적인 파워를 나눌 수 있도록 함
  
    
  - Manifolds
  
    - 원래의 공간보다 더 낮은 차원으로 구성된 manifold를 이용해 매우 작은 타겟 공간을 파악함 (예 - 오토인코더)


  - Natural clustering
   
    - 일반적인 경우와 달리, 서로 연결되지 않은 manifold들의 데이터가 한 클래스에 포함 될 수도 있기 때문에 이러한 형태의 clustering을 고려함
    
    
  - Temporal and spatial coherence
   
    - 대부분의 주요 요소들은 시간, 공간적으로 천천히 변화하거나, 적어도 픽셀 값보다는 이러한 주요 요소들이 예상하기 쉽다는 점을 이용함
    
    
  - Sparsity
  
    - 대부분의 feature들은 목표하는 과제에 사용되지 않는데, 이러한 feature들을 무시함으로써 문제가 쉬워지도록 함
   
    
  - Simplicity of Factor Dependencies
   
    - 요소들 사이의 관계를 단순화 시킴으로써 효과적인 representation을 얻음 (예 - linearity, 확률 독립성)
