import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test




from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
# clear old variables
# 清除旧变量
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
# 定义输入数据（如每轮迭代中都会改变的数据）
# 第一维是None，每次迭代时都会根据输入数据自动设定
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
# 定义模型
def complex_model(X,y,is_training):
    # parameters
    # 定义一些常量
    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPSILON = 0.001

    # 7x7 Convolutional Layer with 32 filters and stride of 1
    # 7x7的卷积窗口，32个卷积核，步长为1
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    # W: 图像宽度   32
    # F：Filter的宽度  7
    # P: padding了多少  0
    # padding='valid' 就是不padding  padding='same' 自动padding若干个行列使得输出的feature map和原输入feature map的尺寸一致
    # S: stride 步长  1
    # (W-F+2P)/S + 1 = (32 - 7 + 2*0)/1 + 1 = 26
    h1 = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='VALID') + bconv1
    # ReLU Activation Layer
    # ReLU激活层
    a1 = tf.nn.relu(h1)  # a1的形状是 [batch_size, 26, 26, 32]
    # Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
    # for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth],
    # 与全局标准化(global normalization)对应，这里的标准化过程我们称之为局部标准化(Spatial Batch Normalization)。记住，我们的卷积窗口大小是[batch, height, width, depth]
    # pass axes=[0,1,2]
    # 需要标准化的轴的索引是 axes = [0, 1, 2]
    axis = list(range(len(a1.get_shape()) - 1))  # axis = [0,1,2]
    mean, variance = tf.nn.moments(a1, axis) # mean, variance for each feature map 求出每个卷积结果(feature map)的平均值，方差

    params_shape = a1.get_shape()[-1:]   # channel or depth 取出最后一维，即通道(channel)或叫深度(depth)
    # each feature map should have one beta and one gamma
    # 每一片卷积结果(feature map)都有一个beta值和一个gamma值
    beta = tf.get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)

    gamma = tf.get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    # mean and variance during trianing are recorded and saved as moving_mean and moving_variance
    # moving_mean and moving variance are used as mean and variance in testing.
    # 训练过程中得出的平均值和方差都被记录下来，并被用来计算移动平均值(moving_mean)和移动方差(moving_variance)
    # 移动平均值(moving_mean)和移动方差(moving_variance)将在预测阶段被使用
    moving_mean = tf.get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = tf.get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # update variable by variable * decay + value * (1 - decay)
    # 更新移动平均值和移动方差，更新方式是 variable * decay + value * (1 - decay)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))


    a1_b = tf.nn.batch_normalization(a1, mean, variance, beta, gamma, BN_EPSILON)
    # 2x2 Max Pooling layer with a stride of 2
    # 2x2 的池化层，步长为2
    m1 = tf.nn.max_pool(a1_b, ksize=[1,2,2,1], strides = [1,2,2,1], padding='VALID')
    # shape of m1 should be batchsize * 26/2 * 26/2 * 32 = batchsize * 5408
    # Affine layer with 1024 output units
    # 池化后的结果m1的大小应为 batchsize * 26/2 * 26/2 * 32 = batchsize * 5408
    # 仿射层共输出2014个值
    m1_flat = tf.reshape(m1, [-1, 5408])
    W1 = tf.get_variable("W1", shape=[5408, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    h2 = tf.matmul(m1_flat,W1) + b1
    # ReLU Activation Layer
    # ReLU激活层
    a2 = tf.nn.relu(h2)
    # Affine layer from 1024 input units to 10 outputs
    # 仿射层有1024个输入和10个输出
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])
    y_out = tf.matmul(a2,W2) + b2
    return y_out


y_out = complex_model(X,y,is_training)

# Now we're going to feed a random batch into the model
# and make sure the output is the right size
# 现在我们随机输入一个batch进入模型，来验证一下输出的大小是否如预期
x = np.random.randn(64, 32, 32,3)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()

        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))