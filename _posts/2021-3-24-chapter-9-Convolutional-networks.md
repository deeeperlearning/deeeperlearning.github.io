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



  Recall the concept of a prior probability distribution from Sec. 5.2. This is a
probability distribution over the parameters of a model that encodes our beliefs
about what models are reasonable, before we have seen any data.
  Priors can be considered weak or strong depending on how concentrated the
probability density in the prior is. A weak prior is a prior distribution with high
entropy, such as a Gaussian distribution with high variance. Such a prior allows
the data to move the parameters more or less freely. A strong prior has very low
entropy, such as a Gaussian distribution with low variance. Such a prior plays a
more active role in determining where the parameters end up.
  An infinitely strong prior places zero probability on some parameters and says
that these parameter values are completely forbidden, regardless of how much
support the data gives to those values.
  We can imagine a convolutional net as being similar to a fully connected net,
but with an infinitely strong prior over its weights. This infinitely strong prior
says that the weights for one hidden unit must be identical to the weights of its
neighbor, but shifted in space. The prior also says that the weights must be zero,
except for in the small, spatially contiguous receptive field assigned to that hidden
unit. Overall, we can think of the use of convolution as introducing an infinitely
strong prior probability distribution over the parameters of a layer. This prior
says that the function the layer should learn contains only local interactions and is
equivariant to translation. Likewise, the use of pooling is an infinitely strong prior
that each unit should be invariant to small translations.
  Of course, implementing a convolutional net as a fully connected net with an
infinitely strong prior would be extremely computationally wasteful. But thinking
of a convolutional net as a fully connected net with an infinitely strong prior can
give us some insights into how convolutional nets work.
  One key insight is that convolution and pooling can cause underfitting. Like
any prior, convolution and pooling are only useful when the assumptions made
by the prior are reasonably accurate. If a task relies on preserving precise spatial
information, then using pooling on all features can increase the training error.
  Some convolutional network architectures (Szegedy et al., 2014a) are designed to
use pooling on some channels but not on other channels, in order to get both
highly invariant features and features that will not underfit when the translation
invariance prior is incorrect. When a task involves incorporating information from
very distant locations in the input, then the prior imposed by convolution may be
inappropriate.
  Another key insight from this view is that we should only compare convolutional
models to other convolutional models in benchmarks of statistical learning
performance. Models that do not use convolution would be able to learn even if
we permuted all of the pixels in the image. For many image datasets, there are
separate benchmarks for models that are permutation invariant and must discover
the concept of topology via learning, and models that have the knowledge of spatial
relationships hard-coded into them by their designer.
