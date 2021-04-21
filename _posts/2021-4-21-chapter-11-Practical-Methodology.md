## 11.5 Debugging Strategies


- 머신 러닝 시스템이 제대로 작동하지 않을 때, 알고리즘 자체의 문제인지 알고리즘의 구현 도중 버그가 발생한 것인지를 확인하기는 쉽지 않음


- 머신 러닝을 이용하는 대부분의 경우에서, 학습을 통해 얻어내려는 알고리즘의 구체적인 사항들은 알지 못함

  - 머신 러닝을 사용하는 이유는 우리가 구체화, 특정화 하지 못하는 패턴 혹은 행동을 얻어내기 위해서임

  - 이 때 만약 어느 과제에 대해 test error가 5% 였다면, 이 수치가 예상된 범주 이내인지 아직 최적화되지 못한 상태인지 알기 힘듬

  - 또한, 대부분의 머신 러닝 모델이 여러 부분으로 구성되어 있는 점도 정확한 원인을 찾아 디버깅하기 힘들게 함


- 따라서 모델의 특정 부분은 그대로 두고, 다른 부분을 변경시키며 테스트하는 방향으로 디버깅 전략이 발전해 옴


  - Visualize the model in action
  
    - 만약 이미지에서 특정 물체를 찾도록 훈련시킨다면, 이 모델이 물체를 찾은 것으로 보고한 이미지를 살펴봄

    - 당연한 소리로 들리겠지만, 단순히 정확도, 성능 등의 수치로만 비교하는 것 보다 더욱 확실히 현실적인 활용성을 검사할 수 있음 


  
  - Visualize the worst mistakes

    - Softmax output layer 등으로 인해 output에 대한 확률이 과대 평가 되었을 수 있음

    - 학습 단계에서 가장 큰 오차를 만든 데이터 샘플을 살펴보며 특징을 파악함

    - 예) Street View에서 주소를 파악할 때, 이미지를 너무 많이 crop해서 주소 일부가 잘렸다면 당연히 큰 오차가 발생할 것임



  - Reasoning about software using train and test error

    - 사용하는 알고리즘이 알맞게 구현되었는지 정확히 확인하기는 어려움

    - 이 때, training error와 test error를 비교함으로써 다양한 원인을 유추해볼 수 있음

    - 만약 training error는 낮은데 test error가 높다면, 학습 과정을 작동하지만 overfitting 되었을 확률이 높음

      - 혹은 모델을 학습시킨 뒤 테스트를 위해 다시 로드하는 과정에서 문제가 생겼을 수 있음

      - 혹은 test data가 training data와 다른 방법으로 가공 되었을 수 있음

    - 만약 training error와 test error 모두 높다면, 단순 소프트웨어 문제인지 알고리즘 자체의 문제인지 확인하기 어렵고, 후술되는 추가적인 확인 과정이 필요함
    


  - Fit a tiny dataset

    - 만약 training error가 높다면, underfitting 상태이거나 소프트웨어에 문제가 있을 것임

    - 한 개의 학습 데이터에 대한 분류를 학습시킨다면, bias를 적절하게 설정하는 것 만으로도 학습이 가능함

    - 만약 한 개(혹은 소수)의 학습 데이터에 대해 학습이 이루어지지 않는다면, 학습 과정에 대한 소프트 웨어 문제일 확률이 높음



  - Compare back-propagated derivatives to numerical derivatives

    - 소프트웨어에서 gradient 계산 과정을 직접 코딩해야 하는 경우 서로 다른 형태의 gradient를 사용해보고, 이 때 방법에 따라 문제가 발생하지 않는다면 적어도 이 부분의 소프트웨어에서 발생하는 문제는 아닐 것임

![_config.yml]({{ site.baseurl }}/assets/ch11/Fig11_diff.PNG)



  - Monitor histograms of activations and gradient
    - Activation 혹은 gradient 값 분포를 살펴보면 모델 상태에 대한 다양한 정보를 파악할 수 있음
    
    - 예) Rectifier가 얼마나 자주 작동하는지, 얼마나 빨리 최적화 상태에 가까워지는지, 얼마나 많은 cell들이 포화되는지 등
