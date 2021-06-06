
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


#### 12.4.3.3 Importance Sampling

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


#### 12.4.3.4 Noise-Contrastive Estimation and Ranking Loss

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

  
#### 12.4.5.1 Using an Attention Mechanism and Aligning Pieces of Data

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


#### 12.5.1.1 Exploration Versus Exploitation

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
