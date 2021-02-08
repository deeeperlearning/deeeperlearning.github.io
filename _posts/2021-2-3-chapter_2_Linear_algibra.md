---
layout: post
title: chapter 2: Linear algibra
---


# chapter 2: Linear algibra

## 2.1 Scalars, Vectors, Metrices, and Tensors

- scalar  $i$: 수, italics로 표기
- vector $\pmb{x}$: 여러 수를 특정 순서로 나열. 굵은 영문 소문자로 표기, 벡터의 각 항은 $x_i$처럼  subscript로 표기
- matrix $A$: 수를 2차원으로 배열한 것. 굵은 영어 대문자로 표현
- tensor **A**: 3차원 이상.
- Transpose(전치): 행렬의 행과 열을 바꾼 것.

### 2.2 행렬과 벡터의 곱셈

- $(AB)^T = B^{T}A^{T}$

### 2.3 단위행렬, 역행렬

- Identity matrix: $\forall x \in \mathbb{R}^{n}, I_{n}x = x$인 $I$
- inverse matrix: $A^{-1}A = I_n$인 $A^{-1}$

### 2.4 Linear dependence and span(생성공간)

- $A^{-1}$이 존재하기 위해서는 모든 b에 대해 $Ax=b$를 만족하는 해가 하나 있어야 한다.
- linear combination:  $Ax = \sum{x_iA_{:, i}}$
- 벡터집합 $\{ v^{(1)}, v^{(2)},... v^{(n)}\}$의 일차결합은 각 벡터에 스칼라 계수를 곱한 것.   
                                              

$\sum_i{c_iv^{(i)}}$
- span: 주어진 벡터 집합의 일차결합으로 얻을 수 잇는 모든 점의 집합
- 일차연립방정식 $Ax=b$의 해가 잇는지는 b가 $A$의 열들의 생성곤간에 속하는지로 판단 가능
- 일차종속: 어떤 벡터집합의 구성 요소의 조합으로 그 벡터집합의 어떤 요소를 만들 수 있는 경우
- 일차독립: 없는 경우

### 2.5 Norms (번역서엔 노름이라 되어있는데 건전하게 놂이라 읽자)

벡터의 크기 구하기. 

- n-th norm:

$$L^p = \parallel{x}\parallel_{p} =   (\sum_{i}|x_i|^p)^{1/p} 
$$

- L2 norm: Euclidean norm, $x^Tx$
- cost func에 많이 쓰는데, 큰, 작은 오차를 얼마나 강조하느냐에 따라 L1, L2 등등을 씀
- max norm: $\max_i|x_i|$ → 이것도 기계학습에서 씀
- 행렬 크기를 구해야 할 때 Frobenius norm(프로베니우스 놂)을 주로 쓴다. L2 norm이랑 비슷

$$||A||_F =\sqrt{\sum_{i,j}{A_{i, j}^{2}}} $$

### 2.6 Special kinds of Matrices and vectors

- Diagonal matrix(대각행렬): $A_{i, j} = 0 \text{ if } i\neq j$
- vector v로 square + diagonal matrix를 만들면 diag(v)로 표기
    - $diag(v)x$는 $v$의 각 성분 $v_i$에 $x_i$배를 한 것.
- symmetric matrix: $A = A^T$
- unit vector: L2 norm의 크기가 1인 vector
- 서로 orthogonal한 두 벡터: $x^Ty=0$
- orthogonal matrix(직교행렬):  각 행들이 서로 정규직교이도 열들도 서로 정규직교인 square matrix
- orthogonal matrix에서는 $A^TA = I$ 따라서 $A^{-1} = A^T$

### 2.7 Eigendecomposition(EVD)

수학적 대상 중에는 그것을 구성요소들로 분해하여 표현 방식과 무관하게 보편적인 어떤 성질을 찾아내면 더 잘 이해할 수 있는 것들이 많다. 

가장 널리 쓰이는 행렬 분해 방법: eigendecomposition(고윳값 분해)

square matrix A의 eigenvector v, eigen value $\lambda$는 아래를 만족한다. 

$$Av = \lambda v$$

$A$의 eigenvector의 집합, 혹은 직교행렬을 V$=\{v^{(1)},...v^{(n)}\}$라고 할 때 A의 Eigendecomposition은 아래와 같이 정의된다. 아래 식은 위 eigenvector들을 한꺼번에 표기한 것이라 보면 된다. 

$$A = Vdiag(\lambda)V^{-1}$$

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled.png)

- 모든 eigen value가 양수인 행렬을 positive definite matrix라고 한다.
- 모든 eigen value가 0 이상인 행렬을 positive semidefinite matrix라고 한다.

### 2.8 Singular value Decomposition(SVD)

- 또다른 방식의 matrix decomposition
- 행렬을 singular vector와 singular value로 분해한다.
- 좀 더 일반적인 행렬들에 적용 가능. 모든 실수 행렬에 적용 가능. square matrix일 필요도 읎음.
- 아래와 같이 어떤 행렬 A가 세 행렬의 곱으로 표현됨

$$A = UDV^T$$

- $A$가 m*n 행렬이면 ***U***는 m*m, ***D***는 m*n, ***V***는 n*n 행렬.
- ***U, V***는 둘 다 orthogonal matrix, ***D***는  diagonal matrix
- ***D***의 main diagonal 성분을 ***A***의 singular value라고 부른다.
- 특이값분해의 기하학적 의미
    - 행렬을 좌표공간에서의 선형변환으로 봤을때,
        - orthogonal matrix의 의미는 회전변환.
        - diagonal matrix의 의미는 작 좌표성분의 스케일 변환
    - $A = UDV^T$ 에서 U, V는 직교행렬, D는 대각행렬이므로 Ax는 x를 먼저 $V^T$에 의해 회전시킨 후 D로 스케일을 변화시키고 다시 U로 회전시키는 것임을 알 수 있다.

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%201.png)

- Thin SVD, compact SVD, Truncated SVD 등도 있음. → 이미지 압축에 사용가능

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%202.png)

Thin SVD

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%203.png)

Truncated SVD

![_config.yml]({{ site.baseurl }}/assets/ch2/Untitled%204.png)

50개의 singular value로 근사한 이미지. 

### 2.9 무어-펜로즈 유사역행렬

- square matrix가 아닌 행렬은 역행렬이 없음.
- 역행렬 비스무리한 연산이 필요할 때 사용 가능.
- $A$의 유사역행렬 $A^+$는 아래와 같이 정의된다.

$$A^+ = lim_{\alpha\rightarrow0}(A^TA + \alpha I)^{-1}A^T$$

- 근데 보통 실제로 유사역행렬을 구할 때에는 고윳값 분해를 사용.

$$A^+=VD^+U^T$$

- U, D, V는 A의 고윳값 분해 결과임.

### 2.10 대각합 연산자(trace operator)

- 대각합 연산자 Tr는 행렬의 모든  main diagonal의 합을 계산한다.

$$Tr(A) = \sum_i{A_{i,i*}}$$

- 여러 수식을 더 단순하게 만듦. 예를 들어 프로베니우스 노름을 아래와 같이 표현 가능

$$||A||_F = \sqrt{Tr(AA^T)}$$

### 2.11 행렬식

- 행렬을 실수 스칼라로 사상하는 함수. Det(A).
- 행렬의 모든 eigen value를 곱한 값과 같다.
- 주어진 행렬을 곱했을 때 공간이 얼마나 확장 또는 축소되는지를 나타내는 측도.
- 행렬식이 0이면 공간은 적어도 하나의 차원에서 완전히 축소됨. (즉 해당 공간에서 부피 0)
- 행렬식이 1이면 공간의 부피가 안 변함.

### 2.12 예시: PCA