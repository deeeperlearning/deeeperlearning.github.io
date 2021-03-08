## 7.11 Bagging and Other Ensemble Methods

- Bagging: 여러 모델을 합쳐 generalization error를 줄이는 방식
- 여러 모델을 독립적으로 학습시키고, 각각의 모델이 text example에 대해 결정을 내림 (vote)
- Model averaging 혹은 ensemble methods이라 불림
- 작동 이유: 서로 다른 모델이 보통 test set에 대해 모두 같은 에러를 만들지는 않기 때문
- 예) k개 regression model이 있고, 모델 $i$의 에러를 $\epsilon_i$라 할 때,
	- variance $E[\epsilon_i^2] = v$, covariance $E[\epsilon_i\epsilon_j] = c$라 하자
	- 모든 model의 평균으로 얻어지는 에러는 $1/k\Sigma\epsilon_i$가 됨

![_config.yml]({{ site.baseurl }}/assets/ch7/Bagging.PNG)

- 모든 모델이 perfectly correlated 되어 $c = v$이면, 위 식의 값은 $v$가 되어, model averaging이 전혀 도움되지 않음
- 모든 모델이 perfectly uncorrelated되어 $c = 0$이면, 위 식은 only $v/k$가 됨
	- 따라서 ensemble은 최소한 각 모델의 성능만큼은 보일 수 있음
	- 각 모델의 error가 서로 independent 할수록 각 멤버보다 우수해짐
	- 예) 서로 다른 모델이 서로 다른 측면을 학습함

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_5.PNG)


## 7.12 Dropout

- Dropout: 많은 수의 neural network로 이루어진 ensemble을 만드는 bagging 방식
- Bagging의 기본 원리는 각각의 test example에 대해 많은 수의 모델을 학습시키고, 평가하는 것임
- 하지만 각각의 model이 큰 네트워크라면, 이를 모두 학습/평가 하기 시간/메모리 측면에서 현실적이지 못함
- Dropout을 이용하면 효율적인 방식으로 여러 네트워크로 구성된 bagged ensemble을 만들 수 있음
- 일반적으로 non-output unit을 제거하는 방식으로 구현함
	- 혹은 구현의 편의를 위해 특정 unit의 output에 0을 곱함

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_6.PNG)

- 각 input/hidden unit이 model에 포함될 확률은 hyperparameter로, 학습 전에 정해짐
	- 일반적으로 input unit의 포함 확률은 0.8, hidden unit은 0.5를 사용

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_7.PNG)

- Inference: 모든 subnetwork member로부터 vote를 수집하는 과정
	- 일반적으로 10~20개의 sub-network (mask)만 이용해도 좋은 성능을 얻음


- Dropout의 일반적인 장점
	- Computationally cheap: per sample per update 마다 n개의 random binary number를 만들고 곱하는 O(n)의 연산만을 요구함
	- 제한 없음: 모델의 종류, 학습 과정에 구애받지 않음
	- 우수성: computationally inexpensive하다고 알려진 다른 regularizer보다 우수한 것으로 알려짐
		- weight decay, filter norm constraints, sparse activity regularization 등


- Dropout의 약점
	- Regularization이므로 모델의 capacity가 감소하기는 해서, 충분히 큰 모델을 사용해야 함
		- 이 때, dataset이나 network가 너무 크면 dropout에 의해 generalization error가 주는 것 보다, 연산량이 늘어나는 효과가 더 클수도 있음
	- Labeled training example의 수가 적을 때는 효과적이지 못한 것으로 알려짐
		- 이러한 조건에선 Bayesian neural network, unsupervised feature learning 등이 더 우수한 것으로 알려짐


- Dropout에 대한 이외의 이야깃거리
	- Linear regression에 적용할 땐, 각 input feature마다 다른 coefficient가 적용되는 $L^2$ weight decay와 동치임이 증명됨
	- 유성 생식에서 서로 다른 개체의 유전자가 조합되며 적응성을 높이듯이, 각 hidden unit들은 다른 특정 hidden unit이 모델에 포함되는지와 상관 없이 잘 작동해야 함
		- 모델끼리 hidden unit을 바꿔도 정상적으로 작동해야 함
	- Dropout의 장점은 input의 raw value의 noise라기보다, input의 information의 noise에 대한 저항성에 가까움
		- 예) 얼굴 인식에서 특정 hideen unit이 '코'를 학습했다면, 이 unit을 없앴을 때 코가 없는 얼굴 이미지도 인식할 수 있음 


