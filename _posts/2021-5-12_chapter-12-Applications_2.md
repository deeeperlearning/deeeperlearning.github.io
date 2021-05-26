## 12.4 Natural Language Processing

- Natural Language Processing (NLP): 인간의 언어를 컴퓨터가 이용할 수 있도록 하는 분야

  - 예시) 기계 번역 - 특정 언어로 쓰인 문장을 입력시키면, 다른 언어로 번역하여 배출

  - 일반적인 신경망 기술을 이용해 다양한 기능이 성공적으로 구현되어 옴

  - 하지만, 높은 성능을 위해서는 일부 domain-specific 전략들이 중요함 (예-순차적 데이터의 처리 방식)

  - 일반적으로 개별 글자나 byte의 배열이 아닌, 단어의 배열로서 데이터를 다룸

  - 모든 단어의 조합은 굉장히 방대하기 때문에, 단어 기반 모델은 굉장히 차원이 높고 sparse한 discrete space에서 작동해야 함



### 12.4.1 n-grams

- ($n$-1)번째까지의 토큰(단어 등)으로부터 $n$번째 토큰의 조건부 확률을 계산하는 모델


![_config.yml]({{ site.baseurl }}/assets/ch12/Eq12_5.PNG)


- Maximum likelihood를 계산하기 위해서, 학습 데이터에서 각각의 가능한 $n$ gram이 몇 번이나 등장하는지를 세기만 하면 되기 때문에, 굉장히 직관적인 모델임

- 따라서 80~90년대에 statistical language modeling 분야에서 핵심적인 모델로 사용되어 옴

- 작은 $n$에 대해서는 고유의 명칭이 있음 - $n$=1: unigram, $n$=2: bigram, $n$=3: trigram, ...

- 일반적으로 $n$-gram과 $n$-1 gram을 함께 학습시켜서, 두 개의 저장된 확률을 이용해 조건부 확률을 계산하기 용이하게 함


![_config.yml]({{ site.baseurl }}/assets/ch12/Eq12_6.PNG)


- 예시) "THE DOG RAN AWAY"를 처리할 때

  - $P$(AWAY | DOG RAN)의 정보가 있다면,
    $P_3$(THE DOG RAN)으로 부터 마지막 단어를 예상할 수 있음


![_config.yml]({{ site.baseurl }}/assets/ch12/Eq12_7.PNG)


- 이 모델의 가장 큰 한계는, 차원의 저주에 의해 대부분의 단어 조합이 training data에 존재하지 않아 확률이 0이라는 것임

  - $P_{n-1} = 0 $이면, $P_n$이 정의 될 수 없음
  
  - 확률에 기본적으로 작은 값을 더해서 사용하거나 (smoothing), 높은 order와 낮은 order의 $n$ gram을 혼용해서 개선해 왔음



### 12.4.2 Neural Language Models

- Neural Language Models (NLM): 단어들의 distributed representation을 이용함으로써 차원의 저주를 해결하고자 함

- $n$-gram models과 달리, 두 단어가 서로 다른 단어임은 인식한채, 동시에 비슷한 단어임을 인식할 수 있음

- Distributed representation은 모델로 하여금 공유하는 특징이 있는 단어들을 비슷하게 인식하도록 함

  - 예) '개'와 '고양이'는 공통점이 많으므로, 문장에서 '개'가 포함된 자리에 '고양이'도 올 수 있음을 예상함

  - 한 문장이 있더라도 정보를 바꿔가며 지수함수적으로 많은 관련 문장들을 만드는 방식으로 차원의 저주를 해소함

- 이 방식을 word embedding이라고도 함 (마치 CNN의 hidden layer에 의한 image embedding처럼)

  - 어휘 수 만큼의 차원에서, 모든 단어들은 한 one-hot 벡터에 대응되어, 모든 단어 사이의 Euclidean 거리는 $\sqrt{2}$ 임

  - 비슷한 문맥에서 자주 등장하거나, 비슷한 특징이 있는 단어들은 서로 가깝도록 embedded됨


![_config.yml]({{ site.baseurl }}/assets/ch12/Fig12_3.PNG)



### 12.4.3 High-Dimensional Outputs

- 많은 언어 인식 앱에서, 글자보다는 단어를 단위로 아웃풋을 생성하고자 함

- 하지만 어휘 수가 많을 때에는, 가능한 경우의 수도 많아져 연산 양이 기하급수적으로 많아짐

- 일차적인 해결 방법으로는 hidden representation을 affine transformation 시켜서 output space을 얻은 뒤, softmax 함수를 적용하면 됨

  - 하지만, 이 affine transformation의 가중치가 어휘 수 만큼의 차원을 가져야 하고, softmax 계산이 모든 아웃풋에 적용되어야 하므로 역시 연산 양은 방대함


#### 12.4.3.1 Use of a Short List

- 초기에는 어휘 수를 1~2만으로 제한하는 방식으로 연산 양 문제를 해결함 (Bengio et al., 2001, 2003)

- 이후 어휘 집합 $\mathbb{V}$을 자주 사용되는 단어의 집합인 shortlist $\mathbb{L}$과 나머지 드문 단어의 집합인 tail $\mathbb{T}$로 나누는 시도가 이루어짐

- 특정 context $C$ 다음의 단어가 드문 단어 그룹 $\mathbb{T}$에 속할 확률은 $P(i\in\mathbb{T}\mid C)$ 임

- 이 때, 해당 단어가 $y$ 일 확률은 아래와 같음

![_config.yml]({{ site.baseurl }}/assets/ch12/Eq12_10.PNG)

  - $P(y=i\mid C,i\in\mathbb{L})$ 은 NLM을, 보다 간단한 $P(y=i\mid C,i\in\mathbb{T})$ 은 n-gram 모델을 이용해서 계산함

- 가장 큰 단점은, NLM을 이용한 일반화에서의 어드밴티지가 매우 활용 빈도가 높아, 곧 의미가 별로 없는 단어에만 적용될 수 있다는 것임

- 따라서 고차원의 아웃풋에 적용할 수 있는 후속 방법들이 등장해 옴


#### 12.4.3.2 Hierarchical Softmax

- 어휘 $\mathbb{V}$ 각각에 대한 확률을 계층적으로 분해하여, 연산 양을 $\mid\mathbb{T}\mid$ 에서 $log \mid\mathbb{T}\mid$ 로 줄일 수 있음

  - 단어의 카테고리를 만들고, 이것의 카테고리를 만들고, 이것의 카테고리를 만들고... 의 방식

![_config.yml]({{ site.baseurl }}/assets/ch12/Fig12_4.PNG)

- 트리의 구조를 최적화하여 연산 양을 최소화하는 것이 이론적으로는 가능하긴 한데, 현실적으로는 힘듬

  - 정보 이론에 기반하여, 최적화된 binary code를 고름

  - 단어당 bit 수가 log(해당 단어의 빈도)와 같도록 트리를 설계하면 됨

  - 하지만 현실적으로는, 아웃풋 계산은 전체 NLM에서 극히 일부분이기 때문에, 줄여봤자 큰 의미가 없음

  - 보다 중요한 과제는 어떻게 일반화된 방법으로 단어의 클래스와 계층을 정하냐는 것임

- 대표적인 장점은 training time과 test time 모두를 줄이는데 효율적임

- 대표적인 단점은 다음에 소개될 sampling-based 방법보다 성능이 낮다는 것임 (아마도 단어 클래스가 제대로 선택되지 않아서)


### 12.4.3.3 Importance Sampling

NLP training을 가속하는 방법중 하나로 현재 문맥의 다음 위치에 나타날 가능성이 매우 적은 단어들에 대해 gradient contribution 계산을 생략하는 것이다. 단어를 일일이 나열해 확률을 계산하는 대신 그 단어들의 부분집합만 추출하는 것이다. 

![chapter%2012%204%20NLP%209c4bec19ceb342de9c7f6d63ac82fbeb/Untitled.png](chapter%2012%204%20NLP%209c4bec19ceb342de9c7f6d63ac82fbeb/12.4nlp01.png)

마지막 softmax output 이전의 hidden layer이 위와 같이 affine transformation이라고 한다면, gradient를 아래와 같이 표기할 수 있다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp02.png)

a는 softmax layer 통과 이전의 벡터(혹은 score)값이며, 이 벡터는 단어 하나당 하나의 성분으로 구성된다. C는 문맥, 그리고 y는 다음에 올 단어를 의미한다. 위 식의 마지막 줄에서 앞의 항은 positive phase라고 부르며 이 항은 $a_y$를 위로 밀어 올린다. 두번째 항은 negative phase라고 부르고 모든 i에 대에 weight P(y=i|c)를 이용해 기울기를 아래로 끌어내린다. negative phase는 하나의 기댓값이므로 montecarlo sampling를 이용해 추정할 수 있다. 그러려면 model 자체에서 sample을 추출해야 하는데, 이는 모든 i에 대해 P(i|C)를 계산하는 비싼 일이다. 

모형 대신 proposal distribution이라는 다른 분포 q에서 분포를 추출할 수도 있다. 이런 분포(보통 unigram, bigram 분포)를 사용함으로서 발생하는 편향은 적절한 weight를 적용해 바로잡으면 된다. 부정적 단어 $n_i$가 추출되었을 때 그에 해당하는 gradient에 적용되는 weight는 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/2.4nlp03.png)

$p_i$= P(i|C)다. 이를 이용해 softmax층의 gradient를 다시 기술하면 아래와 같다. 

![chapter%2012%204%20NLP%209c4bec19ceb342de9c7f6d63ac82fbeb/Untitled%203.png](chapter%2012%204%20NLP%209c4bec19ceb342de9c7f6d63ac82fbeb/12.4nlp04.png)

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp05.png)

bigram distribution


### 12.4.3.4 Noise-Contrastive Estimation and Ranking Loss

큰 사이즈의 vocabulary에 대한 NLP model을 훈련하는 데 필요한 계산 비용을 줄이는 다른 예로 Ranking loss를 들 수 있다. Ranking loss는 각 단어에 대한 NN 모형의 출력을 하나의 score로 간주해 정확한 단어가 다른 단어들보다  더 높은 순위가 되도록 그 단어의 점수 $a_y$를 다른 단어들의 점수 $a_i$보다 높게 책정한다. 그리고 아래 식으로 ranking loss값을 계산한다.  

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp06.png)

관측된 단어의 점수  $a_y$가 부정적 단어의 점수인 $a_i$보다 1 이상 크면 i번째 항에 대한 기울기가 0이 된다. 


### 12.4.4 Combining Neural Language Models with n-grams

NN 기반 model 대비 n-gram model이 가지는 주요 장점은 capacity가 높으면서도 각 sample을 처리하는데 드는 비용이 적다는 것이다. (현재 문맥에 부합하면 소수 tuple만 참조하면 됨) 다시 말해 capacity과 training cost가 서로 독립적이다.  

두 접근 방식을 결함해 계산량을 크게 높이지 않고 capacity를 높이는 것이 가능하다. Bengio 2001, 2003은 NN model과 n-gram model을 ensemble로 구성했다. 두 모델의 오차가 서로 독립이라면 ensemble model을 통해 test error가 감소할 것이다.



### 12.4.5 Neural Machine Translation

어떤 자연어로 된 글/문장을 다른 자연어로 된 같은 뜻의 문장으로 변환하는 task이다. 

Devlin 2014에서는 원본 언어문구 $s_1, s_2...$에 대한 대상 언어 문구 $t_1, t_2...$에 하나의 MLP를 이용해 점수를 부여하는 방법을 사용했다. 이 MLP는 $P(s_1, s_2 ...$|$t_1, t_2...)$를 추정한다. 이 MLP 접근법의 단점은 sequence를 고정된 길이로 만드는 전처리 과정이 필요ㅎ다는 점이다. 좀 더 유연한 input, output을 지원하는 모델이 더 좋을 것 같은데, RNN이 이런 능력을 제공한다. CNN을 사용하는 모델도 있다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp07.png)

  
### 12.4.5.1 Using an Attention Mechanism and Aligning Pieces of Data

Attention mechanism은 machine translation을 위해 만들어진 구조이다. 일반적인 sequence to sequence model은 target language의 길이가 길수록 낮은 성능을 보인다. 이를 극복하기 위해 모델이 중요한 부분만 잘 기억하도록 만드는 것이 Attention mechanism의 핵심 아이디어이다. 

sequence to sequence 모델은 글자, 단어 등의 sequence를 받아 다른 아이템의 sequence를 출력한다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp08.png)

machine translation의 경우 context가 하나의 vector 형태로 전달되는데, 이때 encoder, decoder은 보통 RNN으로 구성된다. seq2seq model에서 RNN은 한 timestep마다 지금 들어온 input과 이전 timestep의 hidden input을 받는다. 이런 모델의 단점은, 긴 input의 경우 전체 맥락을 고정된 context vector로 적절히 나타내기 어렵다는 것이다. 

Attention mechanism은 seq2seq 모델이 decoding 과정에서 현재 스텝과 가장 연관성이 깊은 파트에 집중할 수 있도록 유도하여 이 문제를 해결한다. 예를 들어 '나는 오이와 민트초코를 오이를 좋아해'를 'I love cucumber and mintchoco.'로 번역하는 과정중 decoding 과정에서 "cucumber"이라는 단어를 만들 때 "오이"라는 매칭되는 표현에 더 집중하도록 하는 것이다. 

Attention mechanism에서는 기존 seq2seq model과 달리 모든 step의 hidden state를 decoder에 넘겨준다. 그리고 decoding 단계에서 전체 hidden state를 보고 각 hidden state마다 점수를 매긴다. 이후 매겨진 점수에 softmax를 취해 각 timestep의 hidden state에 곱한다. 그로인해 더 높은 점수를 가진 hidden state는 더 큰 부분을 차지하게 된다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp09.png)

이렇게 점수를 매기는 과정은 decoder가 단어를 생성하는 매 스탭마다 반복된다. 

![_config.yml]({{ site.baseurl }}/assets/ch12/12.4nlp10.png)



### 12.4.6 역사적 관점

NLP에 깔린 아이디어들은 다양한 응용들로 확장되었다. 

- 파싱
- 품사분석
- semantic role labeling
- 다중 과제 학습 아키텍쳐
- t-SNE가 언어 모형 분석 도구로 사용되기도 함.


references

[https://ratsgo.github.io/from frequency to semantics/2017/10/06/attention/](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/10/06/attention/)

[https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)



## 12.5 Other Applications

이번 단원에서는 이전에 다루었던 사물인식, 음성인식 그리고 자연어 처리과정과는 다른 딥러닝 응용기술을 다룬다.


### 12.5.1 Recommender Systems

정보 기술 분야에서 중요한 task중 하나는 온라인상에서 광고나 상품을 추천해주는 시스템이다. 추천 시스템의 발전 초기에는 상품(또는 광고)을 특정 사용자에게 추천해기 위한 최소한의 정보를 사용했다. 예를들어, 사용자 1과 사용자 2가 상품 A, B, C를 모두 좋아할때, 사용자 1이 상품 D를 좋아하면 사용자 2도 상품 D를 좋아할 가능성이 크다. 이러한 원리를 이용한 알고리즘을 collaborative filtering 이라고 한다. 하나의 방법으로, 사용자와 상품을 embedding하여 예측 값을 얻어내는 방법이 있다.

사용자를 $\boldsymbol A$ 행렬의 열에, 상품을 $\boldsymbol B$ 행렬의 행에 임베딩 시키고, $\boldsymbol b$와  $\boldsymbol c$ 를 사용자와 상품의 바이어스 벡터로 두면 예측값은 아래와 같이 나타낼 수 있다.

$$\hat R_{u,i} = b_{u}+ c_{i}+\sum_{j}A_{u,j}B_{j,i}$$

일반적으로는 구해진 예측값 $\boldsymbol{\hat R}$ 과 실제 레이팅 $\boldsymbol R$의 square error를 최소화 시켜서 학습한다.

또 다른 방법으로 SVD를 이용하여 $\boldsymbol R = \boldsymbol{UDV'}$으로 표현하고, 사용자와 상품의 행렬을 $\boldsymbol A = \boldsymbol{UD}$, $\boldsymbol B=\boldsymbol V'$으로 적는 방법도 있다. (실제로 두 방법 모두 Netflix prize에서 좋은 성능을 보여주었다고 한다.)

하지만 이런 collaborative filtering 시스템은 새로운 상품(사용자)이 들어왔을 때 이에 대한 rating이 없기에 다른 상품(사용자)와의 유사성을 계산할 수 없다는 단점이 있다. 이러한 'cold-start recommendation'을 해결하기 위해 사용자 약력 정보나 상품의 특징 정보를 추가적으로 이용한다(content-based recommender systems).


### 12.5.1.1 Exploration Versus Exploitation

추천 시스템을 만들 때, 지도학습에서 강화학습으로 넘어가는 문제가 발생한다.

학습을 위한 데이터를 모으기 위해 인터넷에 들어가면 이미 추천시스템이 적용 된 데이터만 볼 수 있기 때문에 바이어스가 큰 데이터를 얻게되고, 결국 다른 항목을 추천했을 때 발생하는 상황에 대한 정보는 얻지 못한다. 또한 추가적인 데이터를 조심해서 구하지 않으면, 데이터가 많아질수록 추천 시스템은 계속해서 잘못된 결정을 내릴 것이다(옳은 결정은 매우 낮은 확률값을 가지고, 시스템이 옳은 결정을 하지 않는한 학습이 안되기 때문). 따라서 옳은 결정을 할 때만 보상이 주어지는 강화학습과 비슷하다.

강화학습은 탐색과 착취 사이의 균형이 잘 맞아야한다. a라는 행동을 했을 때 1 이라는 보상이 주어진다는 사실을 안다고 할 때, 착취란 행동 a를 실행하여 보상을 얻는 것을 의미하고 탐색이란 어떤 보상이 주어질지 모르는 새로운 행동을 하여 지식을 얻는 과정이다. 어떤 행위자가 보상을 받기까지 긴 시간 기다릴 수 있다면 탐색을, 기다릴 수 없다면 착취를 선택하는것이 합리적일 것이다.


### 12.5.2 Knowledge Representation, Reasoning and Question Answering

딥러닝을 이용한 문장 번역등은 문자나 단어에 대한 임배딩을 사용하면서 굉장히 성공적으로 작동하였다. 이러한 임배딩은 문자나 단어 하나에 대한 개념을 표현한다. 최근에는 단어나 문자 하나에 대한 임배딩을 넘어 관계나 지식 자체에 대한 임배딩 방법을 연구한다 (대표적인 예로 검색엔진).

관계를 딥러닝으로 학습하기 위해서는 관계를 표현하는 학습 데이터와 이를 표현하기에 적절한 모델이 필요하다. 우선 관계를 표현하기에 적합한 학습 데이터의 형태부터 생각해보자.

**관계 학습 데이터**

관계라는 개념을 조금 더 구체적으로 기술하기 위해서는 entity와 관계(또는 지식)을 triplet으로 묶어 표현해야 한다. 아래 예시를 보면 이해가 쉽다.

- 두 entity 사이의 관계를 triplet으로 표현 → (entity1, relation, entity2)
- E.g. (1, is less than, 2)

조금 더 확장해서 어떤 entity 하나에 대한 관계(넓은 의미에서의)를 attribute로 표현할수도 있다.

- tuple로 표현 → (entity, attribute)
- E.g. (dog, has fur)

이런 관계나 성질을 저장하는데에 특수화 된 관계형 데이터베이스들도 이미 많이 개발되어 있다.

- Freebase , OpenCyc, WordNet, Wikibase, ...

**관계 학습 모델**

관계의 학습에는 자연어 처리 모델들이 흔하게 사용된다. 구체적인 예시들은 다음과 같다.

- 언어를 모델링하는것과 지식을 학습하는 것은 굉장히 비슷하므로, 관계형 데이터베이스의 데이터와 자연어 문장을 함께 학습 (Bordes et al., 2011, 2012; Wang et al., 2014a). 또는 여러 관계형 데이터베이스의 데이터를 한꺼번에 학습 (Bordes et al., 2013b).
- Entity들을 벡터로 표현하고 관계는 행렬로 표현하여 학습. 이렇게 하면 관계라는것이 entity에 가해지는 operation처럼 생각할 수 있다 (Bordes et al., 2012).
- Link prediction: 관계형 데이터베이스의 데이터를 이용해 학습한 다음 누락된 관계를 예측하여 새로운 사실을 알아내는 방법 (Wang et al. (2014b), Lin et al. (2015), Garcia-Duran et al. (2015)).
    - 이러한 모델은 평가하기가 굉장히 까다롭다. 만약 어떤 링크가 데이터에는 없는데 모델의 예측으로는 존재해야 한다면, 진짜 없는건지 데이터에 누락된것인지 알기가 어렵다.
    - 한가지 방법은 거짓일 확률이 높은 관계에 대해 모델을 테스트하는 것이다. 예를들어 참인 관계 (entity1, relation1, entity2)에서 entity2를 데이터베이스의 랜덤한 entity로 바꾸면 이 관계는 거짓일 확률이 높다.
- Word-sense disambiguation: 문맥상 단어의 느낌을 추론하는 task도 관계의 학습에 포함된다 (Navigli and Velardi, 2005; Bordes et al., 2012).

궁극적으로는 모델이 reasoning process를 하는것이 가능하고, natural language에 대해 충분한 이해가 있다면 general question answering system을 만들 수 있다. 물론 굉장히 어려운 문제이다.
