import tensorflow as tf

# placeholder: 학습용 데이터를 담는 그릇, 변수에 값을 대입하는 것은 feed를 통해 진행
# None은 크기가 정해지지 않았음을 위미
A = tf.placeholder(tf.float32, [None, 3])
print(A)

# X라는 Placeholder에 넣을 값 생성, 위에 선언한 것 처럼 2번째 차원의 요수 갯수는 3개
a_data = [
    [1, 2, 3],
    [4, 5, 6]
]

# variable: 학습을 통해서 구해야 하는 값으로 그래프를 계산하면서 최적화 된다.(신경망을 좌우하는 값)
# tf.random_normal: 각 변수들을 정규분포 랜덤 값으로 초기화한다.
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

# 입력값과 변수들을 계산할 수식 작성
expr = tf.matmul(A, W) + b

with tf.Session() as sess:
    # 위에서 설정한 Variable 값들을 초기화해야 값들이 할당된다.
    sess.run(tf.global_variables_initializer())
    print(sess.run(expr, feed_dict={A:a_data}))