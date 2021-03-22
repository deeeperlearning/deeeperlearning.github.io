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
