import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

#第一层卷积层尺度和深度
CONV1_DEEP=32
CONV1_SIZE=5

#第二层卷积层尺度和深度
CONV2_DEEP=64
CONV2_SIZE=5

#全连接的节点个数
FC_SIZE=512

def inference(input_tensor,train,regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights=tf.get_variable("weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("biases",[CONV1_DEEP],tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        print(conv1)
        bias=tf.nn.bias_add(conv1,conv1_biases)
        print(bias)
        activated_conv1=tf.nn.relu(bias)

    with tf.variable_scope("layer2-pool1"):
        pool1=tf.nn.max_pool(activated_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights=tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],tf.truncated_normal_initializer(stddev=0.1))#????????NUM_CHANNELS
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
        bias=tf.nn.bias_add(conv2,conv2_biases)
        activated_conv2=tf.nn.relu(bias)

    with tf.variable_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(activated_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    pool_shape=pool2.get_shape().as_list()
    print (pool2.get_shape)#[7,7,1,64]
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope("layer5-fc1"):
        fc1_weights=tf.get_variable("weights",[nodes,FC_SIZE],tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases=tf.get_variable("bias",[FC_SIZE],tf.constant_initializer(0.1))
        if regularizer!=None:
            tf.add_to_collection("loss",regularizer(fc1_weights))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weights=tf.get_variable("weights",[FC_SIZE,NUM_LABELS],tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases=tf.get_variable("bias",[NUM_LABELS],tf.constant_initializer(0.1))
        if regularizer!=None:
            tf.add_to_collection("loss",regularizer(fc2_weights))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit


