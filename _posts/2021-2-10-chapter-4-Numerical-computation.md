- 기계 학습 분야에서는 일반적으로 분석적으로 식을 풀어서 답을 얻는 것이 아니라, 조금씩 조금씩 추정치를 바꾸는 작업을 여러 번 반복하여 문제를 해결한다.
- 대다수의 작업은 선형 수식을 풀거나 최소, 최대값을 찾는 최적화 과정인데, 실수(real number)가 포함되는 계산의 경우 디지털 컴퓨터로 정확히 처리하지 못할 수도 있다.

## 4.1 Overflow and Underflow
- 디지털 컴퓨터로 숫자를 처리하는 이상 rounding error가 발생할 수 밖에 없다.
- underflow: 0에 가까운 숫자가 0으로 처리되어 발생하는 문제. 

  - 예) 의도치않게 분모에 0이 입력되어 계산 오류가 발생 할 수 있다.

- overflow: 큰 숫자가 $\pm\infty$로 처리되어 발생하는 문제

- Example: softmax 함수

  - 소프트맥스 함수의 정의는 아래와 같다. 
    
    $$softmax(x)_i = {exp(x_i) \over \sum^{n}_{j=1}exp(x_j)}$$
    
  - x의 원소가 모두 $c$로 같은 경우 softmax를 취한 값은 $1 \over n$가 되어야 한다. 

  - 하지만 $c$가 크기가 매우 큰 음수인 경우, $exp(x)$ 값을 구할 때 underflow가 발생해 제대로 계산이 이루어지지 않는다. 

  - $c$가 매우 클 경우, $exp(x)$를 구할 때 overflow가 발생해 값이 정의되지 않는다.

  - 이 문제를 해결하기 위해 $z = x - max_i(x_i)$인 $z$ 를 정의하여 $softmax(x)$ 대신 $softmax(z)$ 을 사용하는 방법이 있다.  

  - $log$ $  softmax(x)$를 구현할 때,먼저 softmax를 구하고 log를 취하면 위에서 소개한 문제가 발생할 수 있으니 수치적으로 안정된 방식으로 계산하는 개별 함수를 구현해야 한다. 

## 4.2 Poor conditioning
- Conditioning: 입력 변수의 작은 변화에 대해 함수 값이 얼마나 빨리 변하는지를 뜻한다. 일반적으로 작은 입력값의 변동이 큰 출력의 변화를 가져올 때 문제가 발생할 가능성이 높다. 

- Condition Number

  - $A \in \mathbb{R}^{n\times n}$이 eigenvalue decomposition을 가질 때, 행렬 $A$의 condition number는 아래와 걑이 정의된다. 

  $$condition\ number = max_{i,j}|{\lambda_i \over \lambda_j}|$$

  - 즉, 가장 큰 eigenvalue와 가장 작은 eigenvalue 사이의 비가 condition number이다. 
  - 이 숫자가 큰 행렬은 역행렬을 계산하는 등의 작업을 수행할 때 input의 에러에 대해 민감한 성질을 가진다. 
  - 역행렬을 계산할때의 rounding error 때문이 아니라, 행렬 고유의 성질임. 
  - Poorly conditioned 행렬은 입력 신호 단계에서 발생한 에러를 증폭 시킴

## 4.3 Gradient-Based optimization
- 대부분의 딥러닝 알고리즘은 어떤 함수$f(x)$의 최대값이나 최소값에 대한 x를 구하는 최적화 과정을 포함한다.

- 최적화를 수행할 $f(x)$를 objective function 혹은 criterion, 최소화해야 할때는 cost function, loss function, error function 등으로 부른다. 

- f(x)를 최적화하는 x값은 다음과 같이 *을 병기해 표기한다.  $x^∗ = arg min\ f(x)$

- Gradient descent: x 값을 약간 바꾸었을 때 f(x) 값이 감소한다면, 해당 방향으로 x 값을 수정하는 과정을 반복하여 f(x)의 최소값을 만드는 x를 찾는 방법
  ![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_1.PNG)

- $f’(x) = 0$: critical point, stationary point (local minimum or maximum of saddle point)
  ![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_2.PNG)

- Global minimum: 전체 정의역에 대해 가장 작은 함수값
  - 여러개의 local minima를 갖거나 평평한 saddle point가 많은 경우에는 optimization이 어렵다.
![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_3.PNG)
  
- $f'(x) = 0$인 지점을 critical point라고 부른다. 

- directional derivative in direction $u$: 함수 $f$의 $u$ 방향으로의 기울기
  - 함수 $f$를 최소화하기 위해 $f$를 가장 빠르게 감소시키는 방향을 찾고자 할 때 사용.

  $$min_{u, u^Tu = 1}u^T \bigtriangledown_xf(x) = min_{u, u^Tu = 1}||u||_2 * ||\bigtriangledown_xf(x)||_2cos(\theta)$$

  - $u$에 무관한 텀을 무시하면, $min_u\ cos(\theta)$로 간소화되고, $u$가 gradient와 반대 방향일 때 최소가 됨
  - 위의 방법을 method of steepest descent 혹은 gradient descent라고 한다.

- Gradient Descent 방법을 이용한 minimum 탐색
  - gradient descent를 이용해 새로운 $x$ 값을 찾는 과정은 아래 식과 같다.
  
    $$x' = x-\epsilon \bigtriangledown_x f(x)$$
  
  - 여기서 $\epsilon$ 은 learning rate이며, 한 번에 최대 경사 방향으로 얼마나 많은 거리를 움직일 것인지를 결정한다. 
  
  - Learning rate는 보통 상수를 사용하나, line search라는 방법에서는 몇 가지 learning rate 값을 동시에 테스트해서 $f$를 최소화 시키는 값을 고른다.
  
  - $\bigtriangledown_x f(x)$값이 0에 근접하면 탐색을 종료한다. 

### 4.3.1 Beyond the Gradient: Jacobian and Hessian Matrices
- Jacobian matrix: $f: R^m  \rightarrow R^n$에서 모든 편미분에 대한 행렬

  ![_config.yml]({{ site.baseurl }}/assets/ch4/jacobian.png)

- Second derivative: 미분에 대한 미분으로, curvature를 측정한다고 볼 수 있음

  - -: 아래로 볼록, cost function이 입실론 보다 많이 감소
  
  - 0: curvature가 없음, cost function의 기울기가 1일 때, 입실론 만큼 감소
  
  - +: 위로 볼록, cost function이 입실론 보다 적게 감소
  
    ![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_4.PNG)
  
- Hessian matrix
  - 다변수를 입력 받을 때의 모든 조합을 고려하는 second derivative와 같은 개념의 행렬이며, Hessian matrix의 각 성분은 아래와 같이 정의된다.

    $$H(f)(x)_{i, j} = {\partial^2 \over \partial x_i \partial x_j}f(x)$$

  - Gradient의 Jacobian이라 할 수 있다.

  - 미분의 순서가 바뀌어도 값은 유지되므로, 2차 미분이 continuous하다면 symmetric matrix임

  - Real, symmetric하므로 real eigenvalues와 orthogonal basis eigenvector로 분해 가능

  - $d^{T}Hd$: unit vector $d$방향으로의 2차 미분

  - $f(x)$를 Talor series로 전개할 때, 2차 미분과 Hessian matrix를 이용하여 표현할 수 있다. 이 때, $g$는 gradient이고, $H$는 $x^{(0)}$에서의 Hessian이다. 

    $$f(x)≈f(x)+(x−x^{(0)})^Tg + {1 \over 2}(x−x^{(0)})^T H(x−x^{(0)} )$$

  - minimum point를 찾기 위해 $x^{(0)}$에서 $x$를 gradient $g$ 방향으로 learning rate $\epsilon$ 만큼 이동한다고 했을 때 $x=x^{(0)} - \epsilon g$ 가 된다. 이를 위 Talor series에 대입하면, 

    $$f(x^{(0)} − \epsilon g) ≈ f(x^{(0)}) − \epsilon g^Tg + {1 \over 2}\epsilon^2g^THg.$$ 가 된다.

  - $g^THg$ 가 양수일 때, 테일러 급수 근사를 가장 많이 감소하게 하는 최적의 $\epsilon$ 은 아래와 같다. 

    $$\epsilon^* = {g^Tg \over g^THg}$$

  - 최악의 경우는 $g$가 가장 큰 eigenvalue $\lambda_{max}$와 상응하는 H의 eigenvector의 방향과 일치할 때다.

  - 이때, 최적의 step size는  ${1 \over \lambda_{max}}$가 됨

- Hessian matrix, H를 이용하면 critical point의 성질을 알 수 있음
  - H가 positive definite (모든 eigenvalue가 양수): local minimum
  - H가 negative definite (모든 eigenvalue가 음수): local maximum
  - 양/음인 eigenvalue가 모두 있음: 방향에 따라 min/max 여부가 다름
![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_5.PNG)
  
- Hessian의 condition number가 클 때, gradient descent 방법의 효과가 좋지 않은 예시
  - 한 방향으로는 급격하게 변하고 (예-$\lambda_{max}$에 대한 eigenvector 방향), 다른 방향으로는 조금 변함 (예-$\lambda_{min}$에 대한 eigenvector 방향)
![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_6.png)

- Newton’s method로 이와 같은 문제를 해결 가능
  - second-order Taylor series expansion을 이용해 x(0) 근처의 f(x)를 근사함
  
    ![_config.yml]({{ site.baseurl }}/assets/ch4/newton1.png)
  
  - 위 식을 풀면 아래와 같은 $x^*$을 구할 수 있다. 
  
    ![_config.yml]({{ site.baseurl }}/assets/ch4/newton2.png)
  
  - 함수 f가 positive deﬁnite quadratic 이면Newton’s method는 위 식을 이용해서 한번에 minimum으로 갈 수 있음
  
  - 하지만 현실적으로 f는 국소적으로 positive deﬁnite quadratic이고 전체적으로는 아니므로, 여러번 반복해야 함
  
  - 이와 같이 gradient descent보다 더 빠르게 critical point로 도달할 수 있지만, local minimum 근처 한정이며 saddle point에서는 오히려 안 좋을 수 있음
  
- First-order optimization algorithms의 예시: gradient 만을 이용하는 gradient descent

- Second-order optimization algorithms의 예시: Hessian matrix를 이용하는 Newton’s method

- 함수에 제약을 걸어 성능을 보장하기도 함: 예) Lipschitz continuous 혹은 Lipschitz continuous derivatives를 가지는 함수
  - 변화율이 Lipschitz constant L에 의해 제한되는 함수 $f$
  - <Eq 4.13>
  - 입력의 변화가 작을 때, 출력의 변화가 작을 것이라 보장함
  
- Convex optimization: 강한 제약을 이용해 좋은 성능을 보장함
  - 모든 지점에서 Hessian이 positive semidefinite (eigenvalue가 모두 0 이상)인 confex function에만 적용 가능함
  - 이러한 함수는 saddle point가 없고, local minima가 global minima라 최적화가 용이함

## 4.4 Constrained optimization
- 함수를 최적화 할 때, 모든 x에 대해서 최적화 하는 것이 아니라, 특정 집합 S에 속하는 x에 대해서 최적화 하는 방법
  - S: Feasible points
  - 예시 1: 제약조건을 고려하여 gradient descent를 수정
    - Step size를 정하고 gradient descent step을 만든 후, 결과가 다시 S로 돌아오게끔 projection해줌  
  - 예시 2: Karush-Kuhn-Tucker (KKT)
    - 제약조건이 부등식일 때 적용 가능 (ex-f(x) > 0)
![_config.yml]({{ site.baseurl }}/assets/ch4/fig4_KKT.png)
KKT, Wikipedia

## 4.5 Example: Linear Least Squares
- 자세한 계산 과정 풀이: https://leejunhyun.github.io/deep%20learning/2018/09/27/DLB-04/
