import tensorflow as tf

# 상수 생성
hello = tf.constant('Hello, Tensorflow')
print(hello)


# 출력에서 Tensor(~~)가 출력되는 이유
# Tensorflow의 그래프와 세션이라는 개념을 이해해야 함
a = tf.constant(4)
b = tf.constant(6)
c = tf.add(a, b)
print(c)

# 위에서 정의한 변수들은 정의했을 때 실행되는 것이 아니라 Session 객체와 Run 메소드를 통해 실행
with tf.Session() as sess:
    print(sess.run(hello))
    print(sess.run(c))

