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
