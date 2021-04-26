
좋은 머신러닝 전문가가 되기 위해선 어떤 알고리즘이 존재하고 어떻게 작동 하는지를 정확히 아는 것도 중요하지만, 어떤 알고리즘을 사용할지 정하고 모니터링을 통해 적절한 피드백을 줄 수 있어야한다.

아래의 절차를 참고하면 주어진 문제를 적절하게 해결하는데 도움이 된다.

- 어떤 error metric을 사용할지, target value는 무엇을 사용할지 정한다.
- End-to-end pipeline을 최대한 빨리 구축하고, 적절한 성능 지표를 추정한다.
- 예상보다 성능이 떨어지는 구성 요소가 무엇인지, 그것이 데이터나 소프트웨어의 과적합, 과소 적합 또는 결함 때문인지 진단한다.
- 진단에서 얻은 특정 결과를 바탕으로 새 데이터 수집, 하이퍼 파라미터 조정 또는 알고리즘 변경과 같은 점진적 변경을 반복한다.

## 11.1 Performance Metrics

우리는 error metric을 통해 향후의 행동을 정하기 때문에 error metric을 정하는 것이 첫째로 중요하다. 원하는 성능이 어느정도인지 또한 중요하다. 학계 종사자라면 기대 성능은 기존에 나와있는 벤치마크를 참고하면 되지만, 상업적인 상품을 만드는 사람이라면 비용 효율성, 안정성 등을 고려해야한다. 

- 스팸 메일을 감지하는 기계를 생각해보자. 스팸 메일을 합법적인 메일로 착각하는 것보다, 합법적인 메일을 스팸 메일로 착각하여 쓰레기통에 버리는 것이 cost가 더 클 것이다. 이 경우에는 합법적인 메일를 차단하는 비용이 스팸 메일를 허용하는 비용보다 높은 형태의 총 비용을 사용하는 것이 좋다.
- 레어 이벤트를 1, 일반 이벤트를 0으로 감지하는 이분 분류기를 생각해보자. 정확도를 error metric으로 설정할 경우 이 분류기는 0만 출력해도 매우 높은 성능을 보여준다. 하지만 레어 이벤트를 감지 했는지는 장담할 수 없다. 이 경우 F-score를 사용하는 것이 좋다.

    $$ F = \frac{2pr}{p+r}$$

    $$p = \frac{\text{detected TRUE}}{\text{whole detections of an algorithm}}\ ,\ r = \frac{\text{TRUE detections}}{\text{total number of existing TRUE}}$$

## 11.2 Default Baseline Models

Error metric과 목표설정이 끝나면 어떤 알고리즘을 사용하여 문제를 해결할지 정해야한다. 복잡도가 높지않은 문제라면 굳이 'deep'한 모델을 사용할 필요가 없기 때문에 로지스틱 회귀 모델부터 시작해 보는 것이 좋다.

알맞은 알고리즘을 고르기 위해서 가장 먼저 할 것은 데이터 구조가 어떤 부류에 속하는지 판단해야한다.

길이가 정해진 벡터가 입력으로 들어가는 지도학습 문제라면 fully connected feedforward 네트워크를, 이미지 처럼 토폴로지 구조가 있는 데이터라면 컨볼루션 네트워크를, 시퀀스 데이터라면 gated recurrent 네트워크를 기본 모델로 설정하는 것이 좋다.

이미 연구가 많이 된 문제라면 이미 존재하는 알고리즘을 카피하는 것도 좋은 방법이다.

## 11.3 Determining Whether to Gather More Data

베이스라인 모델이 정해지면 학습 데이터에 대한 성능을 측정하게 될텐데, 만약 성능이 잘 나오지 않는다면 모델 사이즈를 키우거나 여러 파라미터를 조정할 필요가 있다. 그래도 성능이 좋지 않다면 데이터의 퀄리티를 의심해봐야 한다.

학습 데이터에 대한 성능은 잘 나오지만 테스트 데이터에 대한 성능이 좋지 않을 경우에는 데이터 양을 늘리는 것이 효과적인 해결 방법이다.

추가적인 데이터 수집이 어려운 경우에는 모델의 크기를 줄이거나, 가중치 감소 계수와 같은 하이퍼 파라미터를 조정하거나 드롭 아웃과 같은 정규화 전략을 추가하여 정규화를 개선하는 것이 도움이 된다.


## 11.4 Selecting Hyperparameters

대부분의 딥러닝 알고리즘에는 조정해야하는 하이퍼파라미터가 있다. 하이퍼파라미터들은 모델의 성능부터 학습에 필요한 시간이나 메모리까지 영향을 미친다. 이러한 하이퍼파라미터들을 정하기 위한 접근은 크게 두 가지가 있다.

1. 직접 정하기: 조정하려는 하이퍼파라미터에 대한 근본적인 이해 필요
2. 자동으로 정하기: 조정하려는 하이퍼파라미터에 대한 이해가 부족해도 괜찮음. 단, 계산 비용이 큼

### 11.4.1 Manual Hyperparameter Tuning

하이퍼파라미터 튜닝의 가장 큰 목적은 generalization error를 낮추는 것이다. 이를 위해 하이퍼파라미터 튜닝 과정에서는 크게 모델의 effective capacity와 generalization gap을 고려애햐 한다.

- 모델의 effective capacity 조정
    - 모델의 effective capacity에는 모델 자체가 표현 가능한 capacity, 학습 알고리즘이 비용함수를 최적화하는 능력, 비용함수와 정칙화의 선택이 영향을 준다.
    - 보통 모델의 층을 늘리면 모델의 수용력이 커지지만, 학습 알고리즘/비용함수/정칙화 방법에 따라 모델의 실제 수용력이 달라짐을 고려하며 하이퍼파라미터를 튜닝해야 한다.
- Generalization gap 낮추기
    - 보통 generalization error는 U자 모양의 곡선을 따른다. U자 곡선의 왼쪽은 unerfitting, 오른쪽은 overfitting이라 부른다.
    - Underfitting일 때에는 모델의 수용력이 낮고 training error가 높은 상태이다. 이 경우 모델의 수용력을 늘리는 방향으로 하이퍼파라미터를 바꾸어야 한다.
    - Overfitting인 경우 training error와 test error의 차이가 큰 상태이다. 이 경우 적절한 regularization등을 추가하여 generalization gap을 낮추어야 한다.
    - 특정 하이퍼파라미터를 조절해서 U자 곡선의 generalization 상태를 모두 방문할 수 없다는 것을 염두해야 한다. 어떤 하이퍼파라미터는 discrete하거나 binary여서 U자 곡선의 몇 지점만 방문할 수 있다.
- 몇 가지 팁
    - 하나의 하이퍼라미터만 조정할 수 있다면 learning rate를 조정해야 한다. 가장 중요한 하이퍼파라미터이다.
        - Learning rate가 너무 높으면 모델이 발산할 수 있다.
        - Learning rate가 너무 낮은 경우 단순히 모델의 학습 속도가 느려지는 것을 떠나 도달할 수 있는 training error 자체가 높아진다 (일반적으로 비용함수는 convex가 아니기 때문).
    - Test error가 training error보다 높다면 두 가지 중 하나의 방법을 시도해야 한다.
        - Generalization gap이 감소하도록 하이퍼파라미터 조정. 주의할 것은 training error가 증가하는 속도보다 generalization gap이 감소하는 속도가 빨라야 한다는 것.
        - 더 큰 training dataset 확보. Training data가 커서 training error가 감소하면 더 참에 가까운 해를 찾을 확률이 높아짐을 항상 기억해야 한다.

마지막으로 몇가지 대표적인 하이퍼파라미터가 모델에 어떤 영향을 미치는지 정리하면 아래와 같다.

![_config.yml]({{ site.baseurl }}/assets/ch11/14.4.1.png)

### 11.4.2 Automatic Hyperparameter Optimization Algorithms

- 데이터만 준비되면 다른것은 조정할 필요 없이 최적의 모델을 만들어내는 것이 이상적인 학습 알고리즘이다. 하지만 신경망 알고리즘들은 굉장히 많은 하이퍼파라미터들을 가진다.
- 만약 하이퍼파라미터의 좋은 initial condition을 알고 있다면 11.4.1에서 논의한대로 manual 튜닝이 가능하다. 물론 일반적으로는 그런 초기상태를 알기 어렵다. 이러한 문제를 해결하기 위해 하이퍼파라미터 최적화 알고리즘들이 만들어졌다.
- 하이퍼파라미터 최적화 알고리즘들은 validation error같은 값을 최적화해야하는 목적함수로 가지며, 신경망 학습을 감싸는 형태(신경망 학습보다 바깥쪽 반복문 정도로 이해하면 됨)로 구성된다.
- 불행하게도 하이퍼파라미터 최적화 알고리즘에도 하이퍼파라미터가 존재한다. 하지만 이러한 값들은 신경망 자체의 하이퍼파라미터보다 선택하기 쉽다.

### 11.4.3 Grid Search

![_config.yml]({{ site.baseurl }}/assets/ch11/14.4.2.png)

- 튜닝해야 할 하이퍼파라미터가 2~3개인 경우 사용할 수 있는 방법이다. 각 하이퍼파라미터마다 몇 개의 값을 정하고 모든 조합 중 가장 낮은 validation error를 가지는 경우를 선택한다.
- 보통 각 하이퍼파라미터마다 값의 범위를 정하고 log스케일로 서치한다.
    - E.g. $lr \in \{0.1, 0.01, 10^{-3}, 10^{-4}, 10^{-5}\}$
- 보통 그리드서치는 반복적으로 사용할 때 좋은 성능을 보인다. 예를들어 하이퍼파라미터 $\alpha$를 $\{-1, 0, 1 \}$에 대하여 그리드서치를 진행한 경우
    - $a=1$이 가장 좋다고 판단된 경우: $\{1, 2, 3\}$으로 범위를 옮겨서 다시 그리드서치
    - $\alpha=0$이 가장 좋다고 판단된 경우: $\{-0.1, 0, 0.1\}$로 스케일을 줄여서 세밀하게 서치

### 11.4.4 Random Search

![_config.yml]({{ site.baseurl }}/assets/ch11/14.4.3.png)

- 랜덤서치는 그리드서치와 비슷하지만 값을 랜덤하게 샘플링하는 방법이다. 그리드서치에 비해 좋은 값으로 더 빨리 수렴한다.
- 11.4.3의 그림과 비교해보면 어떤 하이퍼파라미터가 결과에 크게 영향을 미치지 않을 때(그림에서 y축), 랜덤서치가 그리드서치에 비해 지수적으로 효율적임을 알 수 있다.
- 랜덤서치를 할 때 주의할 점은 하이퍼파라미터 값을 descretize하거나 binning 하면 안된다는 것이다.

### 11.4.5 Model-Based Hyperparameter Optimization

- 하이퍼파라미터 탐색을 하나의 최적화 문제로도 생각할 수 있다.
- 기울기를 계산할 수 없는 경우 가능한 방법은, validation error를 모델링하고 (아마 validation error를 예측하는 모델을 말하는 듯) 이 모델을 이용해 최적의 하이퍼파라미터를 찾는 것이다 (대부분 bayesian regression model을 사용한다고 함).
- 하지만, 아직까지 이러한 모델기반 하이퍼파라미터 튜닝이 평균적으로 인간 전문가보다 좋다는 연구결과는 없다.
    - 대표적인 문제 중 하나는 하나의 훈련 실험을 끝까지 마친 후에 실험에서 얻은 정보를 활용한다는 점. 인간 전문가의 경우 학습 초반에 뭔가 잘못되고있다는것을 알 수 있다.


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



## 11.6 Example: Multi-Digit Number Recognition

the Street View transcription system의 개발 과정을 소개한다.

- 처리 과정은 자료의 수집에서 시작하며, 수집된 데이터를 휴먼 파워로 labeling한다.
- 초기에 할 일은 performance metrics을 정하는 방법과 어떤 performance metrics을 사용할지 정하는 것이다. 이 프로젝트의 목표는 human level acc인 98%을 달성하는 것이다. 이 acc를 달성하기 위해 coverage를 희생했다. 이 문제에서 coverage는 transcript를 생성할지 말지 결정하는 것에 대응한다.
- sensible baseline system을 정한다. Computer Vision 분야서는 보통 convolutional network with rectified linear units을 사용한다.
- baseline 모델을 튜닝하면서 각 변경사항이 어떻게 성과를 개선하는지 시험했다. 이 프로젝트에서는 coverage와 자료구조에 관한 이해에서 비롯된 것이다. p(y | x) < t이면 input x에 대한 transcription을 거부했다. 이 정의는 모든 softmax output을 곱하는, 임의적으로 선택된 것이다. 이를 개선하기 위해 원칙적인 log-likelyhood를 실제로 계산하도록 output layer과 cost function을 구체화했다. 이런 과정 이후에도 coverage는 90% 미만이었다.
- 성과개선의 병목을 파악하는 수단을 시스템에 넣었다. 문제의 근원이 overfitting인지 underfitting인지 파악하기 위해 training set과 test set의 성과를 측정하는 수단을 넣었다. 확인 결과 두 오차가 비슷했다. 이는 문제의 근원이 underfitting이거나 training set에 결함이 있다는 것이다.
- 모형의 최악의 실패들을 시각화하는 것을 하나의 디버깅 전략으로 추진했다. 그 결과 이미지를 너무 작게 잘라 번지수의 일부 숫자가 누락된 경우 모형이 실수를 하게 됨을 확인했고, 이를 해결하기 위해선 절단 영역을 결정하는 번지수 검출 시스템 정확도를 개선해야했다. 시간을 절약하기 위해 그냥 더 넓은 영역을 잡는 쪽으로 알고리즘을 개선했고 결과적으로 시스템의 coverage가 10% 증가했다.
- 마지막으로 hyperparameter tuning으로 약간의 성능 향상을 이루어냈다.