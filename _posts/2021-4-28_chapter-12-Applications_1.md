
이번 장에서는 컴퓨터 비전, 음성 인식, 자연어 처리 등 다양한 분야에서 응용하고 적용하는 방법을 소개한다. 이번 단원에서 살펴볼 내용은 크게 두 가지이다.

- 인공지능 응용 프로그램에 필요한 대규모 신경망 구현
- 딥러닝을 이용해서 좋은 성과를 낸 구체적인 응용 분야 소개

## 12.1 Large Scale Deep Learning

- 딥러닝은 개별 뉴런은 지능을 가지지 않지만 그러한 뉴런들이 함께 작동하면 지능적인 결과를 도출한다는 점에 기반한다.
- 중요한 점은, 이러한 뉴런이 굉장히 많아야 한다는 것이다. 실제로 최근에 신경망의 성능을 향상시키는데 가장 큰 기여를 한 부분은 신경망의 크기이다.
- 이번 장에서는 이렇게 큰 신경망을 구현하기 위해 어떠한 방법이 필요한지 알아본다.

### 12.1.1 Fast CPU Implementations

GPU가 사용되기 이전에는 CPU를 사용하여 신경망을 구현하였다. 최근에는 GPU를 이용하기 때문에 자세히 알 필요는 없지만, 특정 CPU 제품에 맞추어 계산 과정을 세심하게 조정한다면 계산 속도를 크게 높일 수 있다는 점은 알아두어야 한다. 예를들어 다음과 같은 시도들이 있었다.

- 부동소수점 대신 고정소수점 이용
- 고정소수점 구현을 세심하게 조율
- 자료구조를 최적화해 캐시 적중률을 높임

물론 출시되는 CPU마다 계산 특성이 달라 모든 CPU에 통용되는 이야기는 아니지만, 모델의 크기는 성능과 직결된 부분이기 때문에 간과해서는 안된다.

### 12.1.2 GPU Implementations

GPU는 게임산업의 성장과 함께 빠르게 성장하였다. GPU는 화면에 나타낼 이미지를 표현하는 것에 초점을 맞춘 장치인데, 화면 렌더링 연산은 다음과 같은 특성들을 가지고 있다.

- 개별 계산이 단순함
- 조건에 의한 분기가 거의 없음

따라서 GPU는 단순한 계산을 병렬화 하는데에 초점이 맞추어져 있다.

신경망 알고리즘의 계산은 위에서 소개한 화면 렌더링 계산과 유사하기 때문에 GPU를 신경망 알고리즘의 계산에 사용하게 되었다. 특히, 이를 발전시킨 GPGPU (general purpose GPU; 화면 렌더링에 국한되지 않은 GPU 계산 기술) 기술이 개발되며 GPU를 이용한 신경망 계산이 폭발적으로 증가하였다. 대표적으로 NVIDIA에서 GPGPU 코드를 작성하기 위한 CUDA라는 프로그래밍 언어를 만들었다.

GPGPU 기술이 개발되었다고 해서 모든 문제가 해결된 것은 아니다. GPGPU를 통해 신경망을 효율적으로 구현할 수 있지만, GPGPU를 위한 효율적인 코드를 작성하는 것은 쉬운일이 아니다. 다행히도 이러한 코드들은 잘 모듈화 된 상태이며 지금은 여러가지 신경망 라이브러리등을 통해 GPGPU 코드 구현에 대한 고민 없이 효율적인 신경망을 구현할 수 있다.

### 12.1.3 Large Scale Distributed Implementations

GPU를 이용해 신경망을 구현한다고 해도 한 대의 컴퓨터 자원으로는 굉장히 큰 신경망을 구현할 수 없다. 이러한 한계를 해결하기 위해 훈련과 추론 연산을 여러 컴퓨터로 분산처리하는 방법도 있다.

먼저, 추론 과정을 분산하는 것은 간단하다. 그냥 서로 다른 샘플들을 서로 다른 컴퓨터에서 처리하면 된다. 이를 자료의 병렬성이라 한다.

자료 병렬성 말고 모형의 병렬성도 고려해야 한다. 추론 과정에서는 자료의 병렬성으로 분산처리가 가능했지만, 훈련 과정은 자료 병렬성을 가지기 어렵다. 이러한 제약을 해결하기 위한 한 가지 방법이 비동기 경사 하강법이다.

- 모델의 매개변수들은 하나의 매개변수 서버가 관리
- 여러 컴퓨터에서 매개변수 서버의 매개변수를 덮어쓰며 병렬적으로 신경망 업데이트

비동기 경사 하강법을 사용하면 하나의 하강 단계의 개선 효과는 감소하지만 단위시간동안 처리할 수 있는 하강 단계가 많아져 평균적으로 학습이 빨라진다. 이 방법은 지금도 굉장히 큰 신경망을 학습하는 주된 방법 중 하나이다.

### 12.1.4 Model Compression

상용 프로그램들은 일반인이 사용하기 때문에 작동 시간과 메모리 코스트가 낮아야한다. 그렇기 때문에 기존의 무거운 모델을 대체하는 작은 모델이 필요할 수 있다. 

모델 압축은 오버피팅을 방지하기 위해 모델 사이즈를 키운 경우 적용가능하다. 이 모델들의 사이즈가 커진 이유는 제한된 학습 데이터의 수 때문인데, 이렇게 커진 모델을 가지고 학습 데이터를 늘릴 수 있다면 더 작은 모델로 학습이 가능해진다. 

- 사이즈가 큰 모델은 결국 어떤 함수 $f(\boldsymbol x)$를 학습하게 되는데, 우리가 랜덤하게 $\boldsymbol x$를 뽑아서 학습된 $f$에 넣어주면, 새로운 데이터를 얻을 수 있다.
    - 나중에 사용할 테스트 input의 분포에 맞게 $\boldsymbol x$를 뽑는것이 효율이 좋기에, 기존 training input에 결함을 주거나 generative 모델을 사용하여 $\boldsymbol x$를 구하는 것이 좋다.

처음부터 작은 모델로 학습하는 방법도 있지만, 잘못된 클래스에 대한 사후 분포를 학습할 수 있기 때문에 좋지 않다.

### 12.1.5 Dynamic Structure

데이터 처리 시스템을 가속화 하기 위해서는 입력을 처리하는데 필요한 계산을 설명하는 그래프에 동적 구조를 갖는 시스템을 구축해야 한다. 

- 입력이 주어졌을 때 실행되어야하는 많은 신경망의 하위 집합을 데이터 처리 시스템이 동적으로 결정하게 할수도 있고,
- 개별 신경망이 주어진 정보를 계산할 하위 집합 (hidden units)을 결정하여 내부적으로 동적 구조를 나타낼 수도 있다.

만약 레어한 물체(또는 이벤트)를 찾는 것이 목적이라면 굉장히 정교하고 큰 용량을 가진 분류기가 필요하다. 그러나 여러개의 분류기를 연속적으로 사용하면 더 효율적으로 학습이 가능하다.

1. 작은 용량을 가진 첫 분류기는 높은 recall값을 가지도록, 즉, 레어 오브젝트가 있는 입력을 최대한 버리지 않도록 학습한다.
2. 마지막 분류기는 높은 precision값을 가지도록 학습한다. 

이렇게 일련의 분류기를 사용하게 되면 레어 오브젝트가 없는 입력은 초기에 배제시킬 수 있기 때문에 데이터 처리를 가속화 시킬 수 있다.

또 다른 동적 구조로는 

- 게이터라는 신경망을 사용하여 입력이 주어지면 여러 전문가 네트워크 중 어느 것이 출력을 계산하는 데 사용할 것인지 선택하는 방법.
- 스위치를 사용하여, hidden unit이 문맥에 따라 다른 unit으로부터 입력을 받도록 하는 방법이 있다.

동적 구조화 된 시스템을 사용할 때 한 가지 문제점은 시스템이 서로 다른 입력에 대해 서로 다른 코드 branch를 따르기 때문에, 병렬화가 힘들어 진다는 것이다.

### 12.1.6 Specialized Hardware Implementations of Deep Networks

CPU 및 GPU에 대한 소프트웨어 구현은 일반적으로 부동 소수점 숫자를 나타낼 때 32 또는 64 비트의 정밀도를 사용하지만, inference를 할 때에는 더 낮은 정밀도를 사용해도 된다는 것이 오랫동안 알려져 왔다(숫자당 적은 bit를 사용해도 된다면 하드웨어 표면적이 줄어든다). 또한 GPU가 보여주었듯이 하드웨어의 성능이 딥러닝에 끼치는 영향은 크다.

따라서 일반인이 휴대폰과 같은 저전력 장치에서 딥러닝을 사용할 수 있도록 특수 하드웨어를 구축하는 것은 중요한 일이다.

## 12.2 Computer Vision(CV)

예전부터 가장 활발하게 Deep learning이 연구되는 분야다.

### 12.2.1 preprocessing

원본 데이터가 deep learning을 통한 학습의 input에 적합하지 않을 때가 있다. 이를 preprocessing을 통해 처리하여 학습에 사용한다. CV 분야에서는 이런 preprocessing 과정이 그리 필요하진 않은데, 이미 input data의 모든 픽셀이 0~1 등으로 정규화되어있기 때문이다. 다만 0~1 범위의 데이터와 0~255 범위의 데이터가 섞여있으면 문제가 되기 때문에 범주를 통일하는 preprocessing이 필요할 것이다.

Dataset augmentation도 일종의 전처리라고 할 수 있다. A related idea applicable at test time is **to show the model many different versions of the same input** (for example, the same image cropped at slightly different locations) **and have the different instantiations of the model vote to determine the output.** This latter idea can be interpreted as an **ensemble approach**, and helps to reduce generalization error.→ 데이터 augumentation이 앙상블 방법이라구?!?! 맞음.

training set, test set에 모두 적용 가능한 전처리로 데이터를 더 canonical form으로 만드는 작업이 있다. model이 다루어야 할 데이터의 variation의 양을 줄여주는 역할을 한다. 이 작업은 generalization error 감소에도 도움을 준다. 예를 들어 AlexNet은 각 픽셀에서 training data의 평균을 빼는 전처리를 사용한다.

**12.2.1.1 Contrast Normalization**

가장 안전하게 variation을 줄이는 방법중 하나는 이미지의 contrast를 줄이는 것이다. contrast는 이미지의 밝은 픽셀들과 어두운 픽셀들의 차이가 어느정도인지 말해주는 값이다. contrast를 정량적으로 나타내는 방법은 여러개가 있는데, deep learning에서 많이 쓰이는 방법중 하나는 특정 영역의 standard deviation을 contrast로 사용하는 것이다.

![_config.yml]({{ site.baseurl }}/assets/ch12/Untitled.png)

global contrast normalization(GCN, not graph CNN!)은 다수 이미지의 contrast variation을 줄이기 위해 각 이미지에서 평균 밝기를 뺀 후(zero-mean) 이미지의 standard deviation(contrast)를 특정 값으로 고정하는 방법이다. 이 과정을 통해 전체 픽셀들의 분포가 평균이 0이고 분산이 1인 정규분포로 바뀌게 된다.

zero-contrast에 가까운 이미지: 한 색으로 정렬된 이미지는 이미지에 담긴 정보가 별로 없지만 GCN을 거치면 노이즈가 오히려 강조된다.이런 문제 상황을 고려해 작은 크기의 positive regularization parameter $\lambda$를 도입해 standard deviation에 더하는 방법을 사용하기도 한다. 또한 분모(std)가 특정 값 $\epsilon$ 이상이어야한다는 제약도 둘 때가 있다. $\lambda$는 overflow를 방지하는 장치이기도 하다.

GCN 과정:

![_config.yml]({{ site.baseurl }}/assets/ch12/Untitled2.png)

GCN은, 다른 표현으로, L2 norm을 이용한 비례라고 할 수 있다. GCN을 기술할 때 L2 norm이 아니라 표준편차에 방점을 찍는 이유는, 이미지 크기에 상관 없이 동일한 s를 사용할 수 있기 때문이다.  결론적으로 GCN은 data를 아래와 같이 구면에 mapping한다고 할 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch12/Untitled3.png)

위에서 언급한 "구면에 mapping하는 방법=GCN"은 구면화(sphering)과 별개의 방법이다. sphering은 데이터들이 같은 분산을 가지도록 주성분을 rescaling하는 것이다. sphering은 whitening이라고도 불린다.

![_config.yml]({{ site.baseurl }}/assets/ch12/Untitled4.png)

Global하게 적용된 contrast normalization은 국소적인 특징: 윤곽선이나 모퉁이 등을 강조하지 못할 때가 많다. 이를 극복하기 위해 local contrast normalization 방법이 제안되었다. 이 방법에서는 이미지의 작은 영역들에 대해 contrast normalization을 수행한다. local contrast normalization은 미분 가능한 연산이므로, input preprocessing도 학습의 일부로 사용될 수 있다.

**12.2.1.2 Dataset Augmentation**

data augmentation은 Object recognition에 굉장히 잘 사용된다. Object recognition task에서는 class가 많은 transformations에 대해 invariant하기 때문이다. Augmentation에는 간단한 회전 변환 이외에도 random color perturbation이나 geometric distortion 같은 방법도 사용된다.




## 12.3 Speech Recognition


- Spoken natural language가 포함된 acoustic signal을 이에 해당하는 단어 배열로 변환하는 작업


  - 일반적으로 신호를 20ms 단위로 나눈 벡터를 이용함


- 1980년대에서 2012년까지 주로 hidden Markov models (HMMs)과 Gaussian mixture models (GMMs)이 결합된 시스템을 사용함


  - HMM은 음소의 배열에 대해 모델링

  - GMM은 acoustic feature와 음소 사이의 관계를 모델링


- 최근까지도 automatic speech recognition (ASR) 분야에 GMM-HMM 시스템이 주로 사용되었지만, 신경망이 처음으로 사용되기 시작한 분야 중 하나이기도 함


  - 1990년대에는 TIMIT에 대해 26%의 음소 감지 에러률을 달성 (Garofolo et al., 1993)

  - TIMIT (Acoustic-Phonetic Continuous Speech Corpus): 사물 인식에서의 MNIST와 비슷한 음성 신호 데이터베이스

  - 하지만 GMM-HMM 시스템도 여전히 높은 성능을 보여, 2000년대 후반까지도 신경망은 GMM-HMM이 커버하지 못하는 용도를 찾는 정도로만 쓰였음


- 하지만 더 크고 깊은 신경망의 사용이 가능해지며, 2009년부터는 음성 인식에 대한 비지도 학습에 사용되어 옴


  - Restricted Boltzmann machines (RBM; Part III)을 이용한 비지도 pretraining을 이용해 음소 감지 에러률이 26 $\rightarrow$ 20.7%로 감소 (Mohamed et al., 2009)

  - 이후 다양한 연구들이 진행되며 급격히 발전함

    - 예) TIMIT에서 초점을 둔 단순 음소 인식에서, 많은 양의 데이터에서부터 학습한 단어 사이의 배열에 대한 코딩으로 발전 (Dahl et al., 2012)


- 이후 점차 많은 양의 데이터가 이용 가능해지며, 굳이 비지도 pretraining 없이도 높은 성능을 보이게 됨


  - 특히 CNN을 사용하여 가중치가 시간, 주파수에 대해 중복 적용(replicate)되며, 초기 신경망에서 시간에 대해서만 가중치가 중복 적용 될 때보다 개선됨

  - Input spectrogram을 1차원 벡터가 아니라, 시간과 주파수 축에 대한 2차원 이미지로 처리하여 이미지 학습과 유사한 형태가 됨

  - 또한 deep LSTM RNN이 적용되며 음소 감지 에러률이 17.7%로 감소함.