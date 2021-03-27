얀 르쿤이 1989년에 처음 소개한 Convolutional Neural Networks는 정해진 격자 형태의 자료를 처리하는 데 특화된 신경망이다. 이 장에선 이미지 처리에서 큰 성공을 거둔 CNN의 개념과 사용하는 이유 등을 다룬다. 

## 9.1 The Convolution Operation

![https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Comparison_convolution_correlation.svg/800px-Comparison_convolution_correlation.svg.png](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Comparison_convolution_correlation.svg/800px-Comparison_convolution_correlation.svg.png)

from wikipedia

Convolution 계산식은 아래와 같다. 

$$s(t) = \int x(a)w(t-a)da$$

$s(t) = (x*w)(t)$과 같이 별표로 표기하기도 한다. 위 식에서 x는 input, $w$를 kernel, output $s$는 feature map이라고 부른다. descrete한 경우 위처럼 적분으로 표현하지 않고 아래와 같은 합산 형태로 계산한다. 

$$s(t) = (x*w)(t) = \sum_{a} x(a)w(t-a)$$

convolution은 여러 축에 동시에 적용할 수 있고, commute해서 교환법칙이 성립한다. 

$$S(i, j) = (I*K)(i, j) = (K*I)(i, j) = \sum_m \sum_nI(i-m,j-n)K(m, n)$$

여러 ML 라이브러리에서는 Cross correlation도 convolution의 범주에 포함시킨다. Cross correlation 식은 아래와 같으며 kernel을 뒤집어서 계산하지 않는다는 점에서 Convolution과 상이하다. 

$$S(i, j) = (I*K)(i, j) = \sum_{m} \sum_n I(i+m, j+n)K(m, n)$$

## 9.2 Motivation of CNN

CNN은 ML의 성능과 학습 시스템을 개선하는데 도움을 준 세가지 핵심 개념을 활용한다. 

- Sparse interaction
  kernel의 크기가 input의 크기보다 작기 때문에 sparse weight 혹은 sparse interaction의 성질이 생긴다. CNN에서는 적절한 학습을 통해 큰 img에서 적은 pixel을 가진 kernel으로 특징을 검출할 수 있다. 따라서 적은 수의 parameter만 사용해도 되고, 메모리 사용량도 줄어든다.

![_config.yml]({{ site.baseurl }}/assets/ch9/FUntitled.png)

- paramater sharing
  전통적인 NeuralNet에서는 weight matrix의 각 성분을 한 번씩만 사용하지만, CNN에서는 kernel의 각 성분이 각 입력의 모든 곳에서 사용된다. 아래 그림처럼 $x_i \rightarrow s_i$에 대응하는 weight는 일반적인 NN에서는 한 번만 사용되는 반면, Convolutional layer에서는 모든 i가 같은 weight를 공유한다. 이런 sharing은 필요한 parameter의 수를 줄인다.

!![_config.yml]({{ site.baseurl }}/assets/ch9/FUntitled 1.png)

- equivariant representation
  paramter sharing은 Convolutional layer에 translation equivariance라는 특성을 부여한다. 함수가 equivariance(등변성)하다는 것은 input이 변하면 output이 같은 방식으로 변경된다는 것이다. 함수 f(x), g가 있을 때, $f(g(x)) = g(f(x))$를 만족하면 f(x)는 g 에 대해 equivariance하다. 
  $I$가 정수 좌표에서 이미지의 밝기를 돌려주는 함수, $g$를 mapping function이라고 할 때, $I' = g(I)$가 $I'(x, y) = I(x-1, y)$를 만족한다고 하자. I에 g를 적용한 후 convolution을 적용한 결과와 $I'$에 convolution을 적용한 후 g를 적용한 결과와 같다. 
  Convolution은 inputa에서 특정 feature가 나타나는 위치를 보여주는 2차원 지도를 만든다. 어떤 대상을 이동시킨 input을 넣는다면 output에서도 해당 출력이 같은 양으로 이동한다.
- 힌튼이 capsulnet을 쓸 때 pooling을 대체하기 위한 건데, pixel binning을 하는 건데, input의 변화에 output이 바뀌어버리면 classifier 면에서 안 좋.. ?

## 9.3 Pooling

- 일반적으로 convolutional network의 각 층은 세 단계로 구성됨

  - 첫번째 단계: 평행한 방향의 convolution을 통해 linear activation이 얻어짐

  - 두번째 단계: 각 linear activation이 nonlinear activation function을 거침

    - rectified linear activation function (ReLU) 등이 쓰임; detector stage로도 불림

  - 세번째 단계: Pooling 함수를 이용해 계층의 output을 좀 더 수정함
  
    - Max pooling, 이웃 간의 평균이나 $L^2$ norm, 중앙에서 떨어진 가리에 따른 가중치 등

![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_7.PNG)


- Pooling은 input에 작은 변화가 있을 때 invariant한 output을 만들 수 있도록 함
  
  - 특히 특정 feature가 '어디에 있는지' 보다 '존재 하는지'를 학습할 때 용이함

    - 예) 이미지에 얼굴이 있는지를 탐색할 때, 정확한 위치는 필요 없음

    - 예) input이 오른쪽으로 한 칸씩 밀려도, output 값에는 거의 변화가 없음

![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_8.PNG)


- 서로 다른 목적으로 학습된 output을 pooling하면 특정한 변화에 대한 invariance를 만들 수 있음

  - 예) '5' 이미지의 rotation에 대한 invariance 형성

![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_9.PNG)


- Pooling unit이 detector unit보다 적을수도 있음

  - 예) Output node의 수가 줄어들어 다음 계층에 대한 계산적 부담이 줄어듬

![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_10.PNG)



## 9.4 Convolution and Pooling as an Infinitely Strong Prior

- Prior probability distribution (5.2 복습): 데이터를 갖추기 전, 어느 모델이 reasonable한지에 대한 믿음


- Prior는 probability density가 얼마나 집중되었는지에 따라 약할수도, 강할수도 있음

  - Variance가 큰 Gaussian distribution $\rightarrow$ weak prior

    - Data에 의해 모델이 보다 자유롭게 수정되도록 하며, strong prior에서는 반대임


- Infinitely strong prior는 몇몇 parameter의 확률을 0으로 두어, 절대 사용되지 못하게 함

  - Convolutional network를 infinitely strong prior가 반영된 fully connected network로 볼수도 있음

    - 각 unit에 배정된 작은 receptive field를 제외한 모든 weight는 0이라는 prior

    - 각 계층이 오직 local interaction만을 학습할 수 있다는 가정


- 물론 fully connected network를 만든 뒤 strong prior를 입혀 convolutional network를 만드는건 매우 비효율적임

  - 하지만, 이와 같은 식의 접근으로 convolutional network가 어떻게 작동하는지 좀 더 이해할 수 있음


- 비슷한 맥락에서, convolution과 pooling이 underfitting을 유발할 수도 있음

  - Convolution과 pooling은 prior가 상당히 정확한 경우에만 작동할 수 있음

  - 만약 공간상의 정보를 유지해야 하는 task라면, pooling을 사용함으로써 training error가 늘 수 있음

    - 일부 채널에는 pooling을 사용, 일부에는 사용하지 않음으로서 두 가지 정보를 모두 코딩하는 네트워크 구조가 제안됨 (Szegedy et al., 2014a)

    - 만약 멀리 떨어진 영역 사이의 정보 전달이 필요한 task라면, convolution은 적당한 방법이 아님
## 9.5 Variants of the Basic Convolution Function

- 하나의 커널(kernel)은 오직 하나의 특징만을 뽑을 수 있기 때문에, 신경망에서는 여러 커널을 동시에 사용한다.
- 일반적으로 입력 데이터가 어떤 격자형태의 실수값이 아니라 격자형태의 벡터값(이미지의 경우 RGB에 해당)이기 때문에 입력데이터, 출력데이터 그리고 커널 모두 3-D 텐서인 경우가 많다. (Batch까지 고려한다면 4-D)
    - 4-D 커널 $\mathsf K_{\text{in-channel,out-channel,row,column}}$가 있다고 하면 입력 $\mathsf V_{\text{channel,row,column}}$ 에 대한 출력 $\mathsf Z_{\text{channel,row,column}}$는 다음과 같이 쓸 수 있다.

        $$\mathsf Z_{i,j,k} = \sum_{l,m,n}\mathsf V_{l,j+m-1,k+n-1}\mathsf K_{i,l,m,n}$$

        인덱스에 -1이 붙은 이유는 C나 Python에서는 인덱스가 0으로 시작하기 때문.

- Downsampling convolution

    출력에서 $s$번째 픽셀만 가져오는 방법. ($s$ : stride) 

    모든 픽셀에 대해 컨볼루션을 계산한 후 downsampling하는 것은 코스트가 크기 때문에 아래의 식과 같이 컨볼루션을 $s$번째 픽셀에 대해서만 하는게 좋다.

    $$\mathsf Z_{i,j,k} = c(\mathsf K,\mathsf V, s)=\sum_{l,m,n}\mathsf V_{l,(j-1)\times s+m,(k-1)\times s+n}\mathsf K_{i,l,m,n}$$

    ![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_12.png)

- Zero-padding

    컨볼루션을 하게 되면 출력 데이터는 입력데이터보다 크기가 작아지기 때문에 컨볼루션을 진행하기 전에 입력데이터에 zero-pad를 하는 방법. 

    제로 패딩을 사용하면 커널의 크기, 출력데이터의 크기를 조절하기 쉬워진다.

    - MATLAB에서 제공되는 세가지 패딩 옵션
        - valid

            커널의 모든 픽셀이 입력 이미지에 포함될때 까지만 컨볼루션하는 방법. 출력 이미지의 크기 : m-k+1 (m : input size, k : kernel size)

        - same

            입력과 출력 이미지의 크기가 같아지도록 제로패딩을 하는 방법. 입력 이미지의 테두리 픽셀들의 영향이 작아지는 단점이 있다.

        - full

            컨볼루션이 진행될 때 k번 방문되는 픽셀의 모든 방향으로 제로패딩을 하는 방법. 출력 이미지의 크기 : m+k-1

- Local connection

    컨볼루션과 동일하지만 픽셀마다 가중치(커널)을 다르게 주는 방법. 

    $$\mathsf Z_{i,j,k} = \sum_{l,m,n}\mathsf V_{l,j+m-1,k+n-1}w_{i,j,k,l,m,n}$$

    ![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_14.png)

    Top : local connection or unshared convolution

    Center : convolution

    Bottom : full connection

    각각의 특징(feature)가 이미지에서 작은 영역에 대한 함수일 때 사용.

    또한 입력데이터의 특정 채널과 출력데이터의 특정 채널을 연관시킬 때 사용.

    ![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_15.png)

- Tiled convolution

    특정 주기마다 동일한 커널을 사용하는 방법.

    $$\mathsf Z_{i,j,k} = \sum_{l,m,n}\mathsf V_{l,j+m-1,k+n-1}\mathsf K_{i,l,m,n,j\%t+1,n\%t+1}$$

    % : modulo,  $t$ : the number of different kernels

    ![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_16.png)

    Top : local connection or unshared convolution

    Center : tiled convolution

    Bottom : convolution

- Backprop

    출력 $\mathsf Z$에 대한 기울기 $\mathsf G_{i,j,k} = \frac{\partial}{\partial \mathsf Z_{i,j,k}} J(\mathsf V, \mathsf K)$가 주어졌을 때, 커널 $\mathsf K$와 입력 $\mathsf V$에 대한 기울기는 다음과 같다.

    $$g(\mathsf G, \mathsf V,s)_{i,j,k,l} = \frac{\partial}{\partial \mathsf K_{i,j,k,l}}J(\mathsf V,\mathsf K) = \sum_{m,n}\mathsf G_{i,m,n}\mathsf V_{j,(m-1)\times s +k, (n-1)\times s +l}$$

    $$g(\mathsf K, \mathsf G,s)_{i,j,k} = \frac{\partial}{\partial \mathsf V_{i,j,k}}J(\mathsf V,\mathsf K) = \sum_{\substack{l,m \\ s.t \\ (l-1)\times s+m=j} }\sum_{\substack{n,p \\ s.t \\ (n-1)\times s+p=k}}\sum_{m,n}\mathsf K_{q,i,m,p}\mathsf G_{q,l,}$$

- Bias

    일반적으로 컨볼루션 레이어에서 바이어스는 각 채널마다 하나의 값을 주고 모든 위치에서 같은 값이 적용되도록 하지만, 위치에 따라 서로 다른 바이어스를 줄 수는 경우도 있다. 

    Ex) 제로 패딩을 했을 경우 이미지의 테두리는 적은 양의 정보를 받아오기 때문에 큰 바이어스를 주는 경우가 있다고 한다. 

## 9.6 Structured Outputs

컨볼루션 네트워크는 클래스 라벨이나 회귀를 위한 하나의 값이 아닌 어떠한 텐서 $\mathsf S$를 출력하도록 만들 수도 있다. 예를 들어, 이미지의 픽셀 $(j,k)$가 클래스 $i$에 속할 확률 $\mathsf S_{i,,j,k}$를 내뱉도록 컨볼루션 네트워크를 만들게 되면, 이미지에 포함된 각각의 물체를 구별할 수 있다.

위와같은 pixel-wise labeling을 하는 한가지 방법으로 가중치를 공유하는 recurrent convolutional network가 있다.

![_config.yml]({{ site.baseurl }}/assets/ch9/Fig9_17.png)

이미지($\mathsf X$)가 히든레이어를 거쳐 출력 $\hat \mathsf Y^{(1)}$을 만들어 내고, 다음 단계에서는 이미지($\mathsf X$)와 이전 단계의 출력 $\hat \mathsf Y^{(1)}$가 입력으로 들어가는 형태이다. 

위의 과정을 반복하여 라벨 추측이 끝나게 되면, 인접해있고 동일한 라벨을 가지는 픽셀들을 하나의 물체로 하여 물체의 구역을 나눌 수 있다.

## 9.7 Data Types

- 1-D
    - Single channel

        Audio waveform

    - Multi-channel

        3D 캐릭터의 애니메이션은 시간이 지남에 따라 골격의 포즈를 변경하여 생성하게 되는데, 각 시점에서 캐릭터의 포즈는 캐릭터의 골격에 있는 각 관절의 각도로 표현할 수 있다. 따라서 각 채널은 한 관절의 하나의 축에 대한 각도가 된다.

- 2-D
    - Single channel

        푸리에 변환으로 표현된 audio waveform. 행 : 시간,  열 : 진동수

    - Multi-channel

        컬러이미지

- 3-D
    - Single channel

        CT 스캔 이미지와 같은 의학 이미지

    - Multi-channel

        컬러 비디오

행렬 곱을 기반으로 한 신경망 같은 경우는 상이한 사이즈의 데이터들을 입력으로 받을 수 없지만(입력 데이터의 사이즈가 바뀌면 가중치 행렬의 크기도 바뀌기 때문) 커널을 사용하는 컨볼루션 네트워크는 상관이 없다. 출력 데이터의 크기가 일정하지 않아도 되는 경우에는 일반적인 컨볼루션을 적용하면 된다. 만일 출력 데이터의 크기가 일정해야 한다면 입력데이터의 크기에 비례하여 크기를 조정하는 풀링 레이어를 사용하면 된다.

당연한 얘기지만, "서로 다른 관측으로 얻어졌기 때문에 데이터들의 사이즈가 상이한 경우"에도 컨볼루션 네트워크를 사용할 수 있다는 것은 아니다. 입력 데이터들이 사이즈는 달라도 동일한 관측으로 얻어진 데이터라면 컨볼루션 네트워크를 사용해도 된다는 것이다.

## 9.8 Efficient Convolution Algorithms

- 현재 사용되는 신경망 알고리즘들은 굉장히 많은 수의 유닛으로 이루어져 있다. GPU 병렬처리를 통해 계산을 수행하지만, convolution 계산 알고리즘 자체를 잘 선택하여 계산의 효율성을 높일 수 있다.
- 하나의 방법으로는 Fourier transform(FT)을 이용할 수 있다.
    - Convolution은 커널과 함수를 주파수 영역으로 변환한 후 곱하고 다시 inverse FT 한것과 같다 (항상 같은건 아니고 제약조건이 있긴 하다).
    - 커널과 함수의 크기에 따라 이 방법은 연산량이 낮은 경우가 있다.
- 또 다른 방법으로는 외적을 이용하는 것이다.
    - $d$ 차원의 커널을 $d$개의 벡터들의 외적으로 표현할 수 있는 경우가 있다 (separalbe이라고 부름).
    - 이 경우 $d$ 차원 커널을 이용해 convolution을 하는 것 보다 벡터들을 이용해 1차원 convolution을 한 후 결합하는 것이 더 효율적이다.
- Convolution을 더 빠르게 계산하는 방법이나, 정확도를 떨어뜨리지 않고 근사하는 방법은 활발히 연구되는 주제 중 하나이다.

## 9.9 Random or Unsupervised Features

CNN 역시 학습 과정에 가장 많은 시간이 소요된다. 이 장에서는 이 문제에 대한 해결법 몇 가지를 소개한다.

1. Random initialize
    - 말 그대로 CNN의 커널을 무작위로 형성한다.
    - 실제로 CNN 커널은 무작위로 가중치로 초기화 한 후 pooling에 무작위로 가중치를 부여하면 층들이 자연스럽게 주파수를 선택하고 translational invariant하게 작동하는 것을 확인한 연구도 있다.
2. 사람이 커널 설계
    - 특정 모양을 검출하는 등의 커널을 사람이 직접 설계하는 방법이다.
3. 비지도학습의 판정조건 이용
    - 어떤 연구에서는 작은 이미지 패치들을 k-means clustering으로 클러스터링 한 후 각 클러스터의 무게중심을 커널로 사용했다 (비슷한 이미지들끼리 모은 후 그 모양 검출하는 커널을 썼다는 이야기인 듯).
    - 이렇게 역전파가 아닌 비지도학습의 판정조건을 이용해서 커널을 형성할수도 있다.
    - 3부에서 자세히 설명한다고 한다.
4. 층별로 나누어 학습
    - 역전파를 한꺼번에 하는게 아니라 층별로 하는 방법도 있다. 8장에서 소개한 greedy layer-wise training 처럼 첫번째 층에 분류기를 붙여서 먼저 학습한 후 계산된 feature로부터 다음 층을 학습한다. 이 과정을 반복하여 깊은 모델을 쌓는 방법.
    - 3부에서는 이를 비지도학습의 판정조건을 이용하는 방법으로 확장하여 설명한다고 한다. 이러한 구조의 대표적인 예는 convolutional deep belief networks라고 한다.

## 9.10 The Neuroscientiﬁc Basis for Convolutional Networks

CNN은 생물학적인 뇌의 구조에서 영감을 받았다.

- 고양이의 뇌에서 뉴런이 시각자극에 반응하는 패턴을 보면, 특정 뉴런들은 몇 가지 특이한 이미지 패턴들에 대하여 크게 반응하는 반면 다른 자극에 대하여는 거의 반응하지 않았다.
- 이러한 결과에 영감을 받아 CNN을 만들게 되었다.

조금 더 구체적으로 CNN과 생물학적 뇌의 공통점을 살펴보자. 뇌에는 시각 정보를 본격적으로 처리하기 시작하는 1차 시각 피질(V1)이라는 부위가 있는데, CNN의 설계에는 다음과 같은 V1의 성질을 반영하였다.

- V1은 망막의 이미지 구조를 반영하는 2차원 구조.
- V1의 대부분은 단순세포로 이루어져 있는데, 이 세포들은 이미지의 국소 부분에 대해 거의 선형적으로 활성화된다(어느 정도까지는).
- V1에는 복합세포도 있는데, 이 세포의 활성화는 feature의 작은 위치적 이동에는 불변성을 가진다. 이는 CNN의 pooling layer에 영감을 주었다.

또, CNN의 마지막 층(classifier를 말하는 듯)은 특정 feature에만 반응한다. 실제로 뇌의 하측두피질(IT 피질)에도 특정 개념에만 반응하는 세포들이 있다. 한 가지 예로 할머니와 관련된 자극에만 반응하는 할머니 세포라는 것도 있다.

물론 CNN이 생물의 뇌와 다른점도 있다.

- 사람의 시야는 아주 작은 부분을 제외하고 해상도가 아주 낮다. 눈은 잦은 눈동자 움직임을 통해 얻은 정보들을 합성해 시야 전체가 선명하게 보인다고 믿게 만든다. 반면 CNN은 고해상도 이미지를 통째로 입력받는다.
- 사람의 시각체계는 다른 여러 감각기관과 통합되어 있고 생각 같은 다른 요인들이 영향을 미친다. 하지만 CNN은 아니다.
- 사람의 시각체계는 물체 인식을 제외하고도 여러가지 일을 수행한다. 또, 몸이 세상과 상호작용하는데 필요한 풍부한 3차원 정보를 처리한다. 이러한 구조를 가진 CNN을 사용하는 경우도 있지만 대부분은 아니다.
- V1같은 낮은 level의 영역도 더 높은 level의 영역으로부터의 feedback에 크게 영향을 받는다. 이러한 구조를 CNN에도 적용하려는 연구가 있지만, 큰 성능개선을 이룬적은 없다.
- IT 피질에서 포착하는 feature가 CNN에서 포착하는 feature와 비슷하긴 하지만, 중간층에서도 그러한지 밝혀지지는 않았다. 실제 사람의 뇌에서는 convolution과 pooling 연산을 사용하지 않을지도 모른다.
- 생물학의 뇌에서 시작정보를 처리하는 부분과 CNN의 학습방법은 굉장히 다르다.

실제 뇌의 V1 세포들의 가중치를 가보르 함수로 표현할 수 있다. 이 함수와 CNN 첫 layer의 가중치들을 비교해보면 인식하는 feature들이 비슷하다. 물론 이 사실 하나만으로 사람의 시각 체계를 모델링하는데 CNN이 가장 적합하다고 주장할 수는 없겠지만, 어느정도 작동 원리가 비슷하긴 하다.

## 9.11 Convolutional Networks and the History of Deep Learning

- 딥러닝의 역사에서 CNN은 생물학적인 발견을 잘 모델링 한 대표적인 예이다.
- CNN이 만들어진 후 많은 공모전에서 CNN이 우승했으며 역전파로 훈련해서 잘 작동하게 만든 최초의 신경망이다.
- 물론 일반적인 신경망이 잘 안되는 영역에서 CNN이 잘 작동하는 이유는 밝혀지지 않았다.
    1. 그냥 weight sharing을 하기 때문에 계산 면에서 효율적이기 때문이 수도 있고,
    2. 신경망이 클수록 잘 훈련하기 쉽기 때문일 수도 있다.
- 어찌되었건 CNN은 수십 년 전부터 좋은 성과를 냈으며 grid 구조를 가지는 데이터에 잘 작동하도록 신경망을 특수화하는 좋은 방법이다.
