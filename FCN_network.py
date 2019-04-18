# -×- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io
import cv2,os
import pylab as plt

import TensorflowUtils as utils
import datetime
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224

## vgg 网络部分， weights 是vgg网络各层的权重集合， image是被预测的图像的向量
def vgg_net(weights, image):

    ## fcn的前五层网络就是vgg网络
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}    #字典
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            ## vgg 的前5层的stride都是2，也就是前5层的size依次减小1倍
            ## 这里处理了前4层的stride，用的是平均池化
            ## 第5层的pool在下文的外部处理了，用的是最大池化
            ## pool1 size缩小2倍
            ## pool2 size缩小4倍
            ## pool3 size缩小8倍
            ## pool4 size缩小16倍
            current = utils.avg_pool_2x2(current)  ## 平均池化
        net[name] = current

    return net  ## vgg每层的结果都保存再net中了


## 预测流程，image是输入图像的向量，keep_prob是dropout rate
def inference(image, keep_prob):    #预测
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """

    ## 获取训练好的vgg部分的model
    print("setting up vgg initialized conv layers ...")

    model_data =scipy.io.loadmat(utils.get_config('VGG_PATH'))

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])  #压缩维度

    ## 将图像的向量值都减去平均像素值，进行 normalization
    processed_image = utils.process_image(image, mean_pixel)    #预处理图像

    with tf.variable_scope("inference"):
        ## 计算前5层vgg网络的输出结果
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        ## pool1 size缩小2倍
        ## pool2 size缩小4倍
        ## pool3 size缩小8倍
        ## pool4 size缩小16倍
        ## pool5 size缩小32倍
        pool5 = utils.max_pool_2x2(conv_final_layer)

        ## 初始化第6层的w、b
        ## 7*7 卷积核的视野很大
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        ## 在第6层没有进行池化，所以经过第6层后 size缩小仍为32倍

        ## 初始化第7层的w、b
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        ## 在第7层没有进行池化，所以经过第7层后 size缩小仍为32倍

        ## 初始化第8层的w、b
        ## 输出维度为NUM_OF_CLASSESS
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        ## 开始将size提升为图像原始尺寸
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        ## 对第8层的结果进行反卷积(上采样),通道数也由NUM_OF_CLASSESS变为第4层的通道数
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        ## 对应论文原文中的"2× upsampled prediction + pool4 prediction"
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        ## 对上一层上采样的结果进行反卷积(上采样),通道数也由上一层的通道数变为第3层的通道数
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        ## 对应论文原文中的"2× upsampled prediction + pool3 prediction"
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        ## 原始图像的height、width和通道数
        shape = tf.shape(image)
        ## 既形成一个列表，形式为[height, width, in_channels, out_channels]
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        ## 再进行一次反卷积，将上一层的结果转化为和原始图像相同size、通道数为分类数的形式数据
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        ## 目前conv_t3的形式为size为和原始图像相同的size，通道数与分类数相同
        ## 这句我的理解是对于每个像素位置，根据第3维度（通道数）通过argmax能计算出这个像素点属于哪个分类
        ## 也就是对于每个像素而言，NUM_OF_CLASSESS个通道中哪个数值最大，这个像素就属于哪个分类
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

## 训练
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    ## 下面是参照tf api
    ## Compute gradients of loss_val for the variables in var_list.
    ## This is the first part of minimize().
    ## loss: A Tensor containing the value to minimize.
    ## var_list: Optional list of tf.Variable to update to minimize loss.
    ##   Defaults to the list of variables collected in the graph under the key GraphKey.TRAINABLE_VARIABLES.
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    ## 下面是参照tf api
    ## Apply gradients to variables.
    ## This is the second part of minimize(). It returns an Operation that applies gradients.
    return optimizer.apply_gradients(grads)

class FCN(object):
    def init_net(self):
        ## dropout 的保留率
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        ## 原始图像的向量
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # 1.batch大小，2，H,3,W，4，
        ## 原始图像对应的标注图像的向量
        self.annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")

        ## 输入原始图像向量、保留率，得到预测的标注图像和随后一层的网络输出 logits：未归一化的概率
        self.pred_annotation, self.logits = inference(self.image, self.keep_probability)
        self.trainable_var = tf.trainable_variables()
        self.sess = tf.Session()
        print("Setting up Saver...")
        self.saver = tf.train.Saver()  # 保存
        self.sess.run(tf.global_variables_initializer())  # 初始化所有变量
        ## 加载之前的checkpoint
        MODEL_PATH = utils.get_config('MODEL_PATH')
        self.ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            print("Model restored...")

    def visualize(self,img):
        sp = img.shape
        img = np.reshape(img, [1, sp[0], sp[1], 3])
        prob = tf.nn.softmax(self.logits)
        ans = self.sess.run(prob, feed_dict={self.image: img, self.annotation: np.zeros([1, 1, 1, 1]),
                                        self.keep_probability: 1.0})
        ans = ans[:, :, :, 1] * 255.  # 截取裂缝的channel
        ans = np.round(ans).astype(np.uint8)
        ans = np.squeeze(ans)
        return ans

if __name__ == "__main__":
    tf.app.run()


