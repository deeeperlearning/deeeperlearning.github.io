## 7.11 Bagging and Other Ensemble Methods

- Bagging: 여러 모델을 합쳐 generalization error를 줄이는 방식
- 여러 모델을 독립적으로 학습시키고, 각각의 모델이 text example에 대해 결정을 내림
- Model averaging 혹은 ensemble methods이라 불림
- 작동 이유: 서로 다른 모델이 보통 test set에 대해 모두 같은 에러를 만들지는 않기 때문
- 예) k개 regression model이 있고, 모델 $i$의 에러를 $\epsilon_i$라 할 때,
	- variance $E[\epsilon_i^2] = v$, covariance $E[\epsilon_i\epsilon_j] = c$라 하자
	- 모든 model의 평균으로 얻어지는 에러는 $1/k\Sigma\epsilon_i$

![_config.yml]({{ site.baseurl }}/assets/ch7/Bagging.PNG)

- 모든 모델이 perfectly correlated 되어 $c = v$이면, 위 식의 값은 $v$가 되어, model averaging이 전혀 도움되지 않음
- 모든 모델이 perfectly uncorrelated되어 $c = 0$이면, 위 식은 only $v/k$가 됨
- 따라서 ensemble은 최소한 각 모델의 성능만큼은 보일 수 있고, independent error일수록 각 멤버보다 우수해짐
- 예) 서로 다른 모델이 서로 다른 측면을 학습함

![_config.yml]({{ site.baseurl }}/assets/ch7/Fig7_5.PNG)


## 7.12 Dropout
