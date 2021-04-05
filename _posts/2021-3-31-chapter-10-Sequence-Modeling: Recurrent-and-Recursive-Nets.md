
CNN은 어떤 격자형태의 이미지를 처리하는데 특화된 모델이라면, RNN은 순차적인 데이터(sequential data)를 처리하는데 특화되어있다.

- Parameter sharing

    서로 다른 길이를 가진 데이터들에 대해서도 동일한 모델을 사용할 수 있다. 특히 어떤 특정한 정보가 sequence의 여러 위치에서 나타날 수 있을때 효과적이다.

    예를 들어, "나는 네팔에 2009년에 갔다." 와 "2009년도에 나는 네팔에 갔다." 라는 두 문장에서 네팔에 간 연도를 추출하는 문제를 생각해보자. 기존의 feedforward network는 parameter들이 분리되어 있기 때문에 각 단어가 서로 다른 위치에서 어떤 역할을 하는지에 대해 모든 경우의 수를 학습해야 하지만, parameter를 공유하는 경우에는 그럴 필요가 없다.

    Parameter sharing을 하는 간단한 예로 1-D temporal sequence를 매 타임스텝마다 동일한 커널을 쓰는컨볼루션 네트워크다. 이 경우 출력 데이터의 각각의 노드는 입력데이터의 인접한 몇개의 노드에 대한 함수가 된다. 

    Recurrent network는 조금 다른 방식으로 parameter sharing을 하는데, 출력 데이터의 각각의 노드는 이전 레이어의 출력노드들의 함수가 되고, 각각의 출력 노드는 이전 레이어에서와 동일한 update rule을 통해 구해진다.

## 10.1 Unfolding Computational Graphs

과거 데이터로부터 미래의 데이터를 예측하는 것을 생각해보면, 시간 $t$까지의 모든 데이터($\boldsymbol x^{(t)},\boldsymbol x^{(t-1)},\boldsymbol x^{(t-2)},...,\boldsymbol x^{(1)}$)의 정보를 저장하여 시간 $t+1$ 의 데이터를 예측하는 과정을 생각할 수 있다. 그러나 만약 시간 $t$까지의 데이터를 잘 요약해서 만들어진 일정한 길이를 가지는 벡터 $\boldsymbol h^{(t)}$를 만들 수 있다면 이것을 이용하는 것이 더 효과적일 것이다.

$$\boldsymbol h^{(t)} = f(\boldsymbol h^{(t-1)},\boldsymbol x^{(t)};\boldsymbol \theta)$$

아래 그림은 위의 식을 computational graph로 나타낸 것이다.

![_config.yml]({{ site.baseurl }}/assets/ch10/Fig10_2.png)

왼쪽의 그래프(circuit diagram)은 cycle이 있는 형태이고, 오른편(unfolded computational graph)은 acyclic 그래프이다. 검은 상자는 delay of 1 time step을 의미한다.

왼쪽 그래프를 오른쪽 그래프처럼 각 time step에서의 값을 하나의 노드로 나타내는 방법을 unfolding이라고 한다.

$$\boldsymbol h^{(t)}=g^{(t)}(\boldsymbol x^{(t)},\boldsymbol x^{(t-1)},\boldsymbol x^{(t-2)},...,\boldsymbol x^{(1)})  = f(\boldsymbol h^{(t-1)},\boldsymbol x^{(t)};\boldsymbol \theta)$$

$t$ step 이후의 recurrence는 과거의 모든 시퀀스를 받아오는 함수 $g^{(t)}$로 쓸 수 있지만, unfolding을 하게되면 $g^{(t)}$를 함수 $f$에 대해서 factorize할 수 있게된다. Unfolding을 하면서 얻게되는 이점은 크게 두가지 정도가 있다.

- 더이상 시퀀스 길이에 의존하지 않게 되면서, 학습된 모델은 항상 동일한 입력 사이즈를 갖게된다.
    - Training 데이터에 등장하지 않은 시퀀스 길이를 가진 데이터에 대해서도 일반화가 가능하다.
- 매 타임스텝마다 동일한 파라미터를 가진 함수 $f$를 사용할 수 있다.

## 10.2 Recurrent Neural Networks

다음은 RN의 설계에서 자주 보이는 주요 패턴들이다. 

**1.** 각 time step에서 하나의 output을 출력하며 hidden unit 사이에 recurrent connection이 존재

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.2.png)

- x 값들로 이루어진 input sequence를 $o$ 값들로 이루어진 output sequence로 mapping하는 recurrent network.

**2.** 각 time step에서 하나의 output을 출력하며 한 time step 단계의 출력과 그 다음 time step의 hidden unit 사이에만 recurrent connection이 존재하는 RNN 구조

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.3.png)

- Output layer에서 다음 sequence의 hidden layer로만 recurrent connection이 존재하는 신경망이다. 이 부류의 RNN은 1번에서 소개한 네트워크 구조보다 표현할 수 있는 함수 집합이 적다.(parameter 조합이 N*m*N*M... vs N*1*N*1...) 이 구조에서는 미래로 전달할 수 있는 정보가 output $o$로 한정된다. 이 구조는 덜 강력하지만 훈련이 더 쉬울 수 있다. 각 time step을 다른 time step 구조와 격리해서 다로 훈련할 수 있기 때문이다. 그러면 훈련 과정을 더 높은 수준으로 병렬화할 수 있다.

**3.** Hidden unit 사이에 recurrent unit들이 존재하고, 모든 sequence를 읽어서 하나의 output을 산출하는 신경망

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.4.png)

- 이 recurrence network 구조에서 신경망의 출력은 마지막 node의 값이다. 이런 recurrent network는 추가 처리 과정의 입력으로 사용할 하나의 고정 크기 표현으로 요약할 때 유용하다.

위에서 소개한 패턴중 1번 패턴이 Recurrent network의 대표 구조라고 할 수 있으며 이번 장의 대부분에서 이 패턴이 쓰인다. 이 구조의 x(t) input에 대한 forward propagation 식은 아래와 같다. activation function은 hyperbolic tangent라고 가정하고 output length와 input length의 길이가 같은 경우를 다루겠다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.5.png)

$U$: input to hidden  parameters

$W$: hidden(t-1) to hidden(t) parameters

sequence x에 대한 total loss는 모든 time step에 대한 loss를 더한 것과 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.6.png)

여기서 $p_{model}(y^{(t)} \vert \{x^{(1)}, ...x^{(t)}\})$ 은 model의 output vector $\hat{y}^{(t)}$에서 $y^{(t)}$에 대한 성분으로 주어진다. 이 과정의 학습을 위해선 펼쳐진 그래프에 대한 실행 비용이 linear인 역잔파 알고리즘이 필요한데, 이를 시간 역전파(BPTT)라고 부른다. 

### 10.2.1 Teacher Forcing and network with output recurrence

한 time step의 output과 그 다음 time step에서의 hidden unit 사이에만 recurrent connection이 존재하는 NN의 경우 표현력이 약하다. 이 NN이 제대로 동작하려면 신경망이 미래를 예측하는 데 사용할 과거의 모든 정보를 출력 단위들이 가지고 있어야 한다. 하지만 일반적으로 각 output들은 training set의 target과 비슷하게 가도록 훈련되기 때문에 이 정보를 가질 가능성이 적다. 

이런 단점에도 불구하고 output to hidden 형태의 recurrent network이 사용되는 이유는 다음과 같다. loss function이 시간 t에서의 예측과 t에서의 training target을 비교하는데에 기초하는 경우 hidden to hidden 연결이 없을 때, 모든 time step 단계를 분리해서 각 단계 t에서의 기울기를 개별적으로 계산할 수 있다. 이런 개별 계산은 병렬화가 가능해 빠른 시간에 학습이 가능하다. 

output에서 model의 내부로 들어가는 recurrent connection이 있는 model의 경우 teacher forcing이라는 훈련 기법을 적용할 수 있다. MLE에서 파생된 방법으로 output target인 $y^{(t)}$를 시간 t+1에서의 입력으로 사용한다. time step이 두 단계인 sequence로 예를 들어 MLE를 구하면 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.7.png)

이 모형에서  $t=2$일 때 모형은 sequence x와 training set의 target y가 주어졌을 때 $y^{(2)}$의 conditional probaility를 최대화하도록 훈련된다. 도식은 아래와 닽다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.8.png)

Teacher forcing 방법은 BPTT를 피하는 한 방법이다. hidden to hidden이 존재하는 모형이라도 output to hidden connection이 존재한다면 이 방법을 사용할 수 있다. 

Teacher forcing 방법의 단점은 NN을 open loop mode로 사용할 때(output을 다시 input으로 집어넣는 구조, adversarial method?!) training set과 test set의 차이가 심해질 수 있다는 것이다. 이를 해결하기 위해 teacher forcing & free-running 방법을 병용하는 방법이 있으며, 생성된 값을 입력으로 사용할지 실제 data를 입력으로 사용할지를 무작위로 선택하는 방법도 있다.(bengio et al, 2015b)

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.9.png)

"**adversarial domain adaptation** to encourage the dynamics of the recurrent network to be the same when training the network and when sampling from the network over multiple time steps."

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.10.png)

### **10.2.2 Computing the gradient in a recurrent neural network**

**recurrent** network는 BPTT로 weight를 업데이트한다. parameter $U, V, W, b, c$와 sequence $x^{(t)}, h^{(t)}, o^{(t)}, L^{(t)}$에 대해 t를 index로 하는 node들로 구성된다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.5.png)

loss function의 기울기는 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.11.png)

이 계산은 sequence의 끝에서 거꾸로 진행된다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.2.png)

sequence의 마지막 t=$\gamma$에서의 $h^{(\gamma)}$의 후행노드는 $o^{(\gamma)}$ 뿐이기 때문에 다음과 같이 진행한다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.12.png)

이제 $t=\gamma -1$에서 $t=1$까지 거슬러올라가면서 gradient를 backprop한다. 이때 $o^{(t)}$ 뿐만 아니라 

$h^{(t+1)}$도 $h^{(t)}$의 후행 노드가 된다. 따라서 기울기는 아래와 같이 주어진다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.13.png)

Computational graph의 내부 노드에 대한 gradient를 구한 뒤에는 parameter node에 대한 gradient를 계산하면 되는데, parameter는 여러 단계(time step)들이 공유하므로 미분 연산을 표기할 때 주의가 필요하다. $\nabla_W f$ 연산자는 계산 그래프의 parameter을 공유하는 모든 층의 gadient에 의해 W가 f의 값에 기여하는 정도를 고려해야 한다. 이를 계산하기 위해 dummy variable $W^{(t)}$를 도입한다. $W^{(t)}$는 시간 단계 t에서 W의 값들을 나타낸다. 따라서 $\nabla_{W^{(t)}}$는 단계 t에서 가중치들이 기울기에 기여하는 정도를 나타낸다. 이를 이용해 나머지 parameter들의 gradient를 표기하면 아래와 같다. 

![_config.yml]({{ site.baseurl }}/assets/ch10/10.2.14.png)

### 10.2.3 Recurrent Networks as Directed Graphical Models

확률 분포 관점에서 RNN은 다음과 같이 해석될 수 있다.

- RNN의 output은 확률 분포에서 sampling한 값이며, 타겟하는 확률분포를 cross entropy loss에 결합해 loss 함수를 만든다. 대표적으로 MSE는 RNN의 output이 Gaussian distribution이라고 가정한 것이다.
- 즉, 이전에 등장한 신경망과 동일하게 확률 분포의 log-likelihood를 최적화하는 문제이다.

만약 우리가 만드려는 모델이 과거의 모든 정보를 고려해야 한다면 일반적인 확률 모델의 파라미터는 아래처럼 $O(k^\tau)$개가 될 것이다 ($\tau$: sequence length).

![_config.yml]({{ site.baseurl }}/assets/ch10/10_2_3_fig1.png)

반면 RNN은 latent variable $h$를 도입하여 계산 비용을 크게 줄일 수 있다. 아래 그림처럼 hidden state와 output을 번갈아 계산하면 $O(1)$의 파라미터 개수를 이용해 해결할 수 있다. 즉, sequence length가 늘어남에 따라 모델의 파라미터 개수가 기하급수적으로 증가하지 않는다.

![_config.yml]({{ site.baseurl }}/assets/ch10/10_2_3_fig2.png)

물론 $p(y^{(t)} | y^{(t-1)}, h^{(t-1)})$이 시간에 상관없이 stationary한 경우에만 이러한 parameter sharing이 가능하다. 다른말로하면, 시간$(t-1)$과 시간 $t$의 관계가 시간에 의존하지 않아야 한다.

마지막으로 RNN에서 어떤 값을 sampling하려면 (미래의 값을 예측한다던가 등) 모델에서 샘플을 뽑는 방법을 정해야 한다. 더 구체적으로는 언제까지 모델에서 sampling할지 정해야 한다. 크게 세 가지 방법이 있다.

- Sampling의 끝을 알리는 특수한 문자열을 도입. 예를들어 문장을 출력하다가 <EOS>라는 토큰이 출력되면 샘플링을 중지하는 방법.
- RNN이 베르누이 분포에서 샘플링 한 값을 하나 더 출력해서 (그냥 output node 옆에 classifier 하나 달으라는 의미) 샘플링을 끝낼지 말지 매 스탭 결정하게 하는 방법.
- 모델에 sequence length 자체를 추론하는 구조를 추가하는 방법.

### 10.2.4 Modeling Sequences Conditioned on Context with RNNs

위에서는 sequence $\{y^{(i)}\}$만을 다루는 RNN에 대하여 중점적으로 설명하였다. 하지만 이전 시간의 sampling 값 이외에 추가적인 값이 RNN에 주어지는 경우들도 있다. 수학적으로 표현하자면 RNN이 $p(y|w)$를 모델링한다고 할 때 $w$가 추가적인 input $x$의 함수라고 생각하면 된다.

RNN에 추가적인 input을 넣는 방법은 다음과 같이 크게 세 가지이다.

1. 각 timestamp에 추가 input을 넣는 방법
2. 추가 input을 RNN의 hidden state로 사용하는 방법
3. 1, 2를 함께 사용하는 방법

1번을 그래프로 표현해보면 아래와 같다.

![_config.yml]({{ site.baseurl }}/assets/ch10/10_2_4_fig1.png)

매 timestamp에 추가 input $x$가 주어진다. 이러한 형태의 대표적인 모델은 사진을 input으로 받아 사진을 설명하는 글을 출력하는 RNN이 있다.

물론 아래처럼 매 timestamp에 추가 input의 sequence $x^{(t)}$가 주어지는 경우도 있다.

![_config.yml]({{ site.baseurl }}/assets/ch10/10_2_4_fig2.png)

물론 이러한 형태에는 $x$와 $y$의 sequence 길이가 같아야한다는 제약이 있다. 이러한 제약을 제거하는 방법은 10.4에서 다룬다.


## 10.3 Bidirectional RNNs

- 일반적으로 특정 시점 $t$ 때의 $y$ 값을 계산하기 위해서는 이전의 input만을 사용하지만, 경우에 따라 모든 input을 모두 사용해야 할 수도 있음

  - 예 - speech recognition: 특정 시점의 의미를 정확히 파악하기 위해서는 다음 몇 단어까지 고려해야 함

- Bidirectional RNN: 위와 같은 목적을 위해 고안됨 (아래 그림)

  - $O^{(t)}$: 과거와 미래의 상태를 모두 고려함
  

![_config.yml]({{ site.baseurl }}/assets/ch10/Fig10_11.PNG)

- 위 그림은 1차원 변화(시간)에 대한 도식인데, 2차원 변화(x, y축 방향)로 확장하여 이미지에도 접목할 수 있음



## 10.4 Encoder-Decoder Sequence-to-Sequence Architectures

- RNN이 학습해야 할 input(context)과 output sequence의 크기가 서로 다를 수 있음

- 이러한 경우를 위해 encoder-decoder 혹은 sequence-to-sequence 구조가 고안됨

![_config.yml]({{ site.baseurl }}/assets/ch10/Fig10_12.PNG)

  1) Encoder(reader or input RNN)가 input을 처리해 context $C$를 만듬 (final hidden state)

  2) Decoder(writer or output RNN)이 $C$를 이용해 적절한 크기의 output을 만듬

- 만약 $C$가 벡터라면, decoder RNN은 10.2.4에서 살펴본 vector-to-sequence RNN으로 다양한 구조가 가능함

  - RNN의 initial state로 input을 제공할 수 있음

  - Input이 매 time step에 hidden unit에 직접 연결 될 수 있음

  - 혹은 위의 두 방법이 함께 이용될 수 있음

- 한 가지 명확한 한계로 $C$의 차원이 충분하지 않을 경우 문제가 될 수 있기 때문에, $C$의 크기를 고정하지 않고 유동적으로 바뀔 수 있게하는 방법이 제안됨 (12.4.5.1에서 소개될 예정)



## 10.5 Deep Recurrent Networks

- RNN에서의 대부분의 계산은 아래 3단계로 구분됨

  - Hidden state로 제공되는 input

  - 이전 hidden state에서 다음 hidden state로의 연결

  - Hiden state에서 output으로의 연결

- 각각의 단계는 weight matrix로 구성되며, 그렇기 때문에 한 층으로 구성되는 shallow transformation임

- 이 각각의 단계 자체를, 아래의 그림과 같이 더 deep해지도록 만들수도 있음

![_config.yml]({{ site.baseurl }}/assets/ch10/Fig10_13.PNG)

  (a) Hidden recurrent state

  (b) Deeper hidden-to-hidden part - 다른 time step 사이의 가장-짧은-경로를 길게 할 수 있음

  (c) 가장-짧은-경로 연장 효과가 skip connection에 의해 완화됨



## 10.6 Recursive Neural Networks

- RNN의 chain-like 구조가 아닌, deep tree와 같은 구조의 네트워크

![_config.yml]({{ site.baseurl }}/assets/ch10/Fig10_14.PNG)

- 같은 길이 $\tau$를 가진 sequence에 대해, 깊이(nonlinear operations의 수)가 $\tau$에서 $O(log\tau)$로 감소함

![_config.yml]({{ site.baseurl }}/assets/ch10/recursive.PNG)

- 데이터 구조 자체가 input이 되는 computer vision과 자연어 처리 등에서 널리 사용됨

  - 단, tree 구조가 직접 입력되어야 함

![_config.yml]({{ site.baseurl }}/assets/ch10/NLP.PNG)


