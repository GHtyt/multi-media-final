import tensorflow as tf
from tensorflow import keras as K


class GradientReversal:
    def __init__(self, name="GradRevIdentity"):
        self.call_num = 0             # 用于防止多次调用call函数时, 名字被重复使用
        self.name = name
    
    def call(self, x, s=1.0):
        op_name = self.name + "_" + str(self.call_num)
        self.call_num += 1
        
        @tf.RegisterGradient(op_name)
        def reverse_grad(op, grad):
            return [-grad * s]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": op_name}):   # 将下面的identity的梯度改成op_name对应的梯度计算方式
            y = tf.identity(x)
        return y
    def __call__(self, x, s=1.0):
        return self.call(x, s)

if __name__ == "__main__": 
    # x = tf.placeholder(dtype=tf.float32, shape=(1, 1))  # 不知道为什么, 报错了, 说没有喂入一个shape=[1, 1], dtype=float的值
    x = tf.constant([[1.]], dtype=tf.float32)
    df = K.layers.Dense(1, use_bias=False, kernel_initializer=K.initializers.constant([[1.]]))
    gr = GradientReversal()
    f = df(x)         # f是获得的特征
    f_gr = gr(f, 1)   # 进行梯度反转, 改变参数s的值可以对反转的梯度进行scale, 这一点比较灵活

    dl = K.layers.Dense(1, use_bias=False, kernel_initializer=K.initializers.constant([[10.]]))
    l = dl(f_gr)     # 域分类器使用梯度反转后的特征, 此处只是梯度反转了, 特征ｆ本身的值是不变的

    dy = K.layers.Dense(1, use_bias=False, kernel_initializer=K.initializers.constant([[2.]]))
    y = dy(f)        # 标签分类器使用没有剃度反转的特征

    loss1 = y 
    loss2 = l
    opt = tf.train.GradientDescentOptimizer(0.1)
    op = opt.minimize(loss1 + loss2)

    # print(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("f, l, y, x:", end=" ")
        [print(x, end=" ") for x in sess.run([f, l, y, x])]
        print("\nfw, lw, yw:", end=" ")
        [print(x[0], end=" ") for x in sess.run([df.weights, dl.weights, dy.weights])]
        sess.run(op)
        print("\nfw, lw, yw:", end=" ")
        [print(x[0], end=" ") for x in sess.run([df.weights, dl.weights, dy.weights])]