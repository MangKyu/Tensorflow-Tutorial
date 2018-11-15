import tensorflow as tf

# 입력값 X
x_data = [1, 2, 3]

# 입력에 대한 정답값 Y
y_data = [2, 4, 6]

# 입력과 정답 값을 담기 위한 placeholder
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 가중치와 편향을 선언하고, 정규분포의 랜덤값으로 초기화
W = tf.Variable(tf.random_uniform([1]))
B = tf.Variable(tf.random_uniform([1]))

# 우리가 계산하는 값이 나올 함수 => 행렬이 아니므로 tf.matmul을 사용하지 않음
h = W * X + B

# 손실 함수를 작성, 예측한 값과 실제 값의 제곱의 평균을 구하는 reduce_mean
cost = tf.reduce_mean(tf.square(h - Y))

# Gradient Descent 함수를 통해 손실 함수가 최소가 되는 W, B를 구한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# 비용을 최소화 하는 것이 목표이다.
# 정의한 optimizer를 통해 정의한 cost가 최소가 되도록 계산해준다.
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 1000번 수행한다.
    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(B))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(h, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(h, feed_dict={X: 2.5}))
