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

  - Hidden state에서 output으로의 연결

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
