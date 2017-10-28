# coding=utf-8
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from zhouchengshuju import train_set,train_label,test_set,test_label

class SparseAutoEncoder():
    def __init__(self, n_input, n_hidden,learningRate=0.01,l2_penalty=0.0):
        #初始化参数
        self.inputDim=n_input #输入数据维数
        self.hiddenDim = n_hidden #隐藏层单元个数
        self.activation=self.relu #激活函数，默认sigmoid
        self.learningRate=learningRate
        self.displayStep=10
        self.alpha=l2_penalty
        self.beta=0
        self.rho=0.05
        self.momentum=tf.Variable(0.9,tf.float32)
        self.corruptionRate=0.0
        # 初始化编码器权重、阈值
        self.encoderWeights = self.InitWeights((self.inputDim, self.hiddenDim))
        self.encoderBiases = self.InitWeights((1, self.hiddenDim))
        # 初始化解码器权重、阈值
        self.decoderWeights = tf.transpose(self.encoderWeights)
        self.decoderBiases = self.InitWeights((1, self.inputDim))
    def relu(self,x):
        greater=tf.greater(x,1)
        smaller=tf.less(x,0)
        zeros=tf.zeros_like(x)
        ones=tf.ones_like(x)
        xx=tf.where(greater,ones,x)
        xxx=tf.where(smaller,zeros,xx)
        return xxx
    def GetWeights(self):
        return (self.encoderWeights,self.encoderBiases)
    def GetHiddens(self):
        return(self.hidden)
    def InitWeights(self,shape):
        # return tf.Variable(tf.random_normal(shape))
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    def Encode(self,X):
        l = tf.matmul(X, self.encoderWeights) + self.encoderBiases
        return self.activation(l)

    def Decode(self,H):
        l=tf.matmul(H,self.decoderWeights)+self.decoderBiases
        return self.activation(l)

    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(tf.clip_by_value(rho,1e-10,1.0)) - rho * tf.log(tf.clip_by_value(rho_hat,1e-10,1.0))+ \
               (1 - rho) * tf.log(tf.clip_by_value((1 - rho),1e-10,1.0)) - (1 - rho) * tf.log(tf.clip_by_value((1 - rho_hat),1e-10,1.0))
    def corruption(self,X):
        return(tf.nn.dropout(X,1-self.corruptionRate))
    def Sparseloss(self,X):
        H = self.Encode(self.corruption(X))#去噪
        rho_hat=tf.reduce_mean(H,axis=0)
        kl=self.kl_divergence(self.rho, rho_hat)
        X_=self.Decode(H)
        diff=X-X_
        cost1=0.5*tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2),reduction_indices=[1]))
        # cost2=self.beta*tf.reduce_mean(kl)
        # regularizer = tf.contrib.layers.l2_regularizer(self.alpha)
        # cost3=regularizer(self.encoderWeights) + regularizer(self.decoderWeights)
        # cost= cost1+cost2+cost3
        return cost1
    def Sparsetraining(self,trainingData, n_iter=500,batch_size=90):
        n_samples, n_dim = trainingData.shape
        X=tf.placeholder("float",shape=[None,n_dim])
        total_batch = int(n_samples/ batch_size)
        cost=self.Sparseloss(X)
        optimizer = tf.train.MomentumOptimizer(self.learningRate,self.momentum).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_iter):
                for i in range(total_batch):
                    batch_xs = trainingData[i*batch_size:(i+1)*batch_size-1]
                    _, c = sess.run([optimizer,cost], feed_dict={X: batch_xs})
                if (epoch+1) % self.displayStep == 0:
                    print("Epoch:", '%04d' % (epoch + 1),"cost=", "{:.9f}".format(c))
            print("Optimization Finished!")
            self.hidden = sess.run(self.Encode(X), feed_dict={X: trainingData})

n_inputs = 200
n_hidden_1 = 100
n_hidden_2=50
n_hidden_3=15

sae = SparseAutoEncoder(n_inputs, n_hidden_1,learningRate=0.01)
sae.Sparsetraining(train_set,n_iter=300)
W1,b1=sae.GetWeights()
hidden1=sae.GetHiddens()
sae2 = SparseAutoEncoder(n_hidden_1,n_hidden_2,learningRate=0.01)
sae2.Sparsetraining(hidden1,n_iter=300)
W2,b2=sae2.GetWeights()
hidden2=sae2.GetHiddens()
sae3 = SparseAutoEncoder(n_hidden_2,n_hidden_3,learningRate=0.01)
sae3.Sparsetraining(hidden2,n_iter=300)
W3,b3=sae3.GetWeights()
hidden3=sae3.GetHiddens()

training_epochs = 300  # 训练批次
batch_size = 90  # 随机选择训练数据大小
displayStep = 5  # 展示步骤
# fan_in=1/math.sqrt(n_hidden_3)
l2_penalty=0.0001
momentum=tf.Variable(0.95,tf.float32)
corruptionRate=0.10

X = tf.placeholder("float", [None, n_inputs])
# weights = {'h1': W1,'h2': W2,'h3': W3,'h4':tf.Variable(tf.random_uniform([n_hidden_3,4],-fan_in,fan_in))}
# biases = {'b1': b1,'b2': b2,'b3': b3,'b4': tf.Variable(tf.random_uniform([4],-fan_in,fan_in))}
weights = {'h1': W1,'h2': W2,'h3': W3,'h4':tf.Variable(tf.truncated_normal([n_hidden_3,4],stddev=0.1))}
biases = {'b1': b1,'b2': b2,'b3': b3,'b4': tf.Variable(tf.truncated_normal([4],stddev=0.1))}


def relu(x):
    greater = tf.greater(x, 1)
    smaller = tf.less(x, 0)
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    xx = tf.where(greater, ones, x)
    xxx = tf.where(smaller, zeros, xx)
    return xxx
#衰减学习率初始化
globalStep=tf.Variable(0,dtype=tf.float32)
learning_rate=tf.train.exponential_decay(0.8,globalStep,train_set.shape[0] / batch_size,0.99)
def Encode(x):
    layer_1 = relu(tf.add(tf.matmul(x, weights['h1']),biases['b1']))
    layer_2 = relu(tf.add(tf.matmul(layer_1, weights['h2']),biases['b2']))
    layer_3 = relu(tf.add(tf.matmul(layer_2, weights['h3']),biases['b3']))
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']),biases['b1']))
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['b2']))
    # layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']),biases['b3']))
    return layer_3
def corruption(X):
    return(tf.nn.dropout(X,1-corruptionRate))
y_pred = tf.nn.softmax(tf.matmul(Encode(corruption(X)),weights['h4'])+biases['b4'])
y_true = tf.placeholder(tf.float32,[None,4])
# diff=y_pred-y_true
regularizer=tf.contrib.layers.l2_regularizer(l2_penalty)
cost=0.5*tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)),reduction_indices=[1]))+regularizer(weights['h1'])+regularizer(weights['h2'])+regularizer(weights['h3'])
# cost = 0.5*tf.reduce_mean(tf.pow(diff,2))+0.5*l2_penalty*(tf.nn.l2_loss(weights['h3'])+tf.nn.l2_loss(weights['h2'])+tf.nn.l2_loss(weights['h1']))
    #-tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)),reduction_indices=[1]))+0.5*l2_penalty*(tf.nn.l2_loss(weights['h3'])+tf.nn.l2_loss(weights['h2'])+tf.nn.l2_loss(weights['h1']))
optimizer = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(cost,global_step=globalStep)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(train_set.shape[0] / batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs = train_set[i*batch_size:(i+1)*batch_size-1]
            batch_ys = train_label[i*batch_size:(i+1)*batch_size-1]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,y_true:batch_ys})
        if (epoch+1) % displayStep == 0:
            print("Epoch:", '%04d' % (epoch + 1),"cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    correct_prediction=tf.equal(tf.argmax(y_true,1),tf.arg_max(y_pred,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accuracy.eval({X:test_set,y_true:test_label}))
    print(sess.run(y_pred,{X:test_set,y_true:test_label}))


