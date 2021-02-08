Ian Goodfellow Deeplearning책의 시작. 

## 1-0 개괄

- What is Machine learning, Deep learning

인공지능의 진정한 난제는 사람들이 쉽게 해내지만 형식적으로 서술하기 어려운 과제들을 푸는 것. 즉, 인간의 직관에 의해 풀 수 있는 문제들을 풀어내는 것. 

이 문제들의 해결책, 다른 말로 "컴퓨터가 개념들의 계통구조(hierachy of concept)를 이용해 경험으로부터 배우고 세상을 이해하는 방법"을 찾는 것.

계통구조에서 각 개념은 자신보다 더 간단한 개념들과의 관계를 통해 정의된다. 간단한 개념들을 재조합하는 더 복잡한 개념을 배운다. 그 개념의 연결관계를 그래프로 표현하면 여러 층으로 이루어진 깊을 그래프로 나오기 때문에 이런 인공지능 접근 방식을 심층학습(Deep learning)이라 부른다. 

- Representation, Feature

로지스틱 회귀와 같은 간단한 기계학습 모델들은 주어진 자료의 표현(representation)에 크게 의존한다. AI 시스템이 환자를 직접 진찰하는 것이 아니라 의사가 시스템에 여러 정보(자궁 반흔 여부 등)을 제공한다. 환자의 표현에 있는 그런 정보 조각들을 특징(feature)이라고 부른다. 로지스틱 회귀를 예로 들면, 이 ML 모델은 이런 특징과 그 결과의 상관관계를 학습하지만 특징들이 정의되는 방식에는 전혀 영향을 주지 않는다. (즉 hand-crafted feature을 사용한다는 이야기) MRI 사진과 같은 raw한 데이터로부터 직접 뭔가를 예측할 수는 없다.(MRI 개별 픽셀이 갖는 정보로 label을 예측할 수는 없음)

이런 형태의 모델에서 자료를 어떻게 표현(representation)하느냐가 알고리즘의 성능을 좌우한다. 어떤 특징을 데이터로부터 추출해야 할지 알아내기 어려운 과제들도 많다.

그럼 어떻게 하냐? 표현과 output(label)을 mapping하는 방법 뿐만 아니라 representation을 뽑아내는 방법 자체도 학습 알고리즘으로 배우게 하면 된다. 이런 접근 방식을 representation learning이라고 한다. 

representation learning의 대표적인 예로 Autoencoder가 있다. Autoencoder는 입력자료를 "다른 표현"으로 변환하는 Encoder 부분과 만들어진 "다른 표현"을 원래 입력자료 형태로 바꾸는 Decoder로 구성되어 있다. Autoencoder은 encoder→decoder을 거치는 과정에서 정보가 최대한 남아있도록 훈련된다.  이 과정에서 잘 이루어진다면, "다른 표현"은 입력자료의 특성을 잘 담고있게 된다.   

Feature learning을 위한 알고리즘을 만들 때에는 변동 인자(factor of variation)을 추출하는 것을 목표로 둔다. factor of variation는 관측된 자료를 설명해주는 어떤 요인이다. : 자료에 존재하는 어떤 변동성(variability)를 이해하는데 도움이 되는 어떤 개념, 추상. 예를 들어 녹음된 음성에서는 화자의 나이, 성별, 억양 등등. 

인공지능 응용의 난제는 관측 가능한 자료에 영향을 주는 변동인자가 너무 많다는 것. 예를 들어 자동차의 윤곽선이라는 factor은 시선 각도에 따라 다 달라진다. 따라서 변동인자를 풀어해쳐서 주어진 과제와 무관한 변동인자를 골라내는 능력이 필요하다. 예를 들어 뇌파에 더해져있는 EOG 같은 것?

representation learning이 주어진 문제 자체를 푸는 것보다 어려운 경우도 있다. 

딥러닝에서는 자료를 좀 더 간단한 표현들을 이용해 표현함으로써 표현학습의 인자가 많은 문제를 해결함. 아래처럼 모서리가 edge들이라는 좀 더 간단한 개념들로 구성됨. 

![_config.yml]({{ site.baseurl }}/assets/ch1/Untitled.png)

심층학습은 학습이 깊게, 여러 층으로 이루어지기 때문에 컴퓨터가 다단게 프로그램을 배울 수 있다. 이런 층별 실행 방식은 다음 층의 명령들이 이전 층의 명령들의 결과를 참조할 수 있으므로 강력하다. 한 층에서 활성화된 정보 중 입력을 설명하는 변동 인자들(factor of variation→ 이거 latent variable이라고 할 수 있나?)을 부호화하지 않는 것들도 있을 것이다. 

- 심층 학습의 깊이

심층학습 모형의 깊이를 측정하는 두 방법이 있다. 

1. 구조 평가를 위해 실행해야 하는 순차 명령의 수. flow chart에서 가장 긴 경로의 길이. 
2. 개념들의 관계를 나타낸 그래프의  깊이를 모형의 깊이로 간주. 
3. 요즘은 parameter 개수로 많이 세지 않나?

- 정리

심층학습은 기계학습의 일종으로 컴퓨터 시스템이 경험과 자료를 통해 자신을 스스로 개선하게 하는 기법. 

## 1-1 이 책의 대상 독자

1부는 기본 수학 도구, ML 개념, 2부는 유우명한 심층 학습 알고리즘, 3부는 앞으로 유망할 모험적인 주제들. 

## 1-2 심층 학습의 역사적 추세

한줄요약: 데이터가 늘어나 더 유용해짐, 하드웨어 기반이 발전해 더 모형이 커짐, 지금 많이 쓰이는 중. 

### 1.2.1 신경망의 다양한 이름과 흥망성쇠

- neuroscience and deep learning

cybernetics → connectoinism+neural network 

뇌에서 따왔다 → artificial neural network라는 이름을 얻었다. 생명체의 뇌는 잘 기능하니 이를 reverse engineering하여 그 기능을 인공적으로 구현하려 하려고 했다. 

현세대의 Deep learning은 이런 신경과학적 관점을 뛰어넘은 것이다. Deep learning은 "multiple levels of composition, 다수준 구성"을 일반화한 것. 신경과학에 없는 구조도 많이 채용된다. 

왜 근래의 DL 연구들에서 신경과학의 역할이 적어졌을까? 우리가 뇌를 이해하지 못하기 때문이다.구조적으로 너무 복잡하고 거대해서 뇌 네트워크 연구가 지지부진하다. (sea elegans 같은 건 가능한데? 저자의 이유보단, 쉽계 계산 가능한 computational structure의 한계때문이 아닐까함. 병렬분산처리가 용이한 SW, HW 연산과 real neural network의 HW topology가 다르징.)

물론 뇌를 계산적으로 연구하는 computational neuroscience 분야는 계속 발전중. 

- 분산 표현(distributed representation)

"시스템의 각 입력을 여러개의 특징으로 표현해야 하며, 각 특징이 가능한 많은 입력의 표현에 관여해야 한다"

예를 들어, 차&트럭&새를 인식할 수 있는 시각 시스템에서 각 물체의 색은 빨,주,파 일 수 있다. 

이 경우 조합은 빨간트럭, 빨간새, 녹색트럭.. 등등이다. 이 시스템 입력을 잘 표현하는 방법으로 각 조합에 대해 대해 개별적인 뉴런 혹은 은닉 단위를 활성화하는 방법이 있다. 그러면 총 조합 개수의 뉴런이 필요하며, 각 뉴런이 색상과 물체의 조합 특성을 독립적으로 학습해야 한다.

distributed representation을 사용하면 이를 더 가볍게 만들 수 있다.   색상을 서술하는 뉴런과 물체의 종류를 서술하는 뉴런만 있으면 된다. 

이 책의 핵심 개념임

- backpropagation

요즘 이거 안 쓰는 데 없음. 이것도 connectionist movement의 산물임

- sequential model

LSTM\[Hochreiter et al, 1997\] 

- 암흑기

딥러닝 코인은 90년대에 크게 좋은 성과를 내지 못했고, 다른 ML 방법론들, 예를 들어 kenel machine\[Boser elt al 1992\], graphical model\[jordan, 1998\] 등이 대두되며 2007년까지 침체됨. 

- 다시 떡상

2006년, 힌튼옹의 Deep belief network에서 greedy layer-wise pretraining을 통해 효율적으로 학습이 가능함을 보임.

### 1.2.2 Increasing Dataset Size

  

![_config.yml]({{ site.baseurl }}/assets/ch1/Untitled%201.png)

빅데이터의 대두! → 양질의 데이터가 큰 사이즈로 마련되니 학습의 부담인 "통계적 추정(statistical estimation)"의 부담이 줄어들었다. 일반화가 쉬워졌다. 

### 1.2.3 모형의 크기 증가

![_config.yml]({{ site.baseurl }}/assets/ch1/Untitled%202.png)

![_config.yml]({{ site.baseurl }}/assets/ch1/Untitled%203.png)

### 1.2.4 Increasing accuracy, complexity, and real-world impact

![_config.yml]({{ site.baseurl }}/assets/ch1/Untitled%204.png)