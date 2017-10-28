import tensorflow as tf
import mnist_inference
import os
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=100
LEARNING_RATE=0.8
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINGING_STEPS=30000
MOVING_AVERAGE_RATE=0.99

MODEL_PATH="/path/to/model/"
MODEL_NAME="model_5.5.ckpt"

def train(mnist):
    with tf.name_scope("input1"):
        x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
        y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-input")
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    with tf.name_scope("moving_average"):
        variable_average=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_RATE,global_step)
        variable_average_op=variable_average.apply(tf.trainable_variables())

    with tf.name_scope("loss_function"):
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        loss=cross_entropy_mean+tf.add_n(tf.get_collection("loss"))
    with tf.name_scope("train_step"):
        learning_rate=tf.train.exponential_decay(LEARNING_RATE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_average_op]):
        train_op=tf.no_op(name="train")

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer=tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
        for i in range(TRAINGING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            if i%1000==0:
                run_option=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata=tf.RunMetadata()
                _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys},options=run_option,run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata,"step%03d"%i)
                print ("After %d training step(s) ,loss on training batch is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_PATH,MODEL_NAME),global_step=global_step)
            else:
                 _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
                 saver.save(sess,os.path.join(MODEL_PATH,MODEL_NAME),global_step=global_step)
    writer.close()
def main(argv=None):
    mnist=input_data.read_data_sets("/tem/data/",one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()
