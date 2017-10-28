import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
mnist=input_data.read_data_sets("/path/to/mnist",one_hot=True)

learning_rate=0.001
training_steps=100000
batch_size=128
n_inputs=28
time_steps=28

hidden_size=128
n_layers=2
n_classes=10

x=tf.placeholder(tf.float32,[None,n_inputs,time_steps])
y=tf.placeholder(tf.float32,[None,n_classes])
weights={"in":tf.Variable(tf.random_normal([time_steps,hidden_size])),"out":tf.Variable(tf.random_normal([hidden_size,n_classes]))}
biases={"in":tf.Variable(tf.constant(0.1,shape=[hidden_size])),"out":tf.Variable(tf.constant(0.1,shape=[n_classes]))}

def RNN(X,weights,biases):
    X=tf.reshape(X,[-1,time_steps])
    X_in=tf.matmul(X,weights["in"])+biases["in"]
    X_in=tf.reshape(X_in,[-1,time_steps,hidden_size])

    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*n_layers)

    state=cell.zero_state(batch_size=batch_size,dtype=tf.float32)
    output,final_state=tf.nn.dynamic_rnn(cell,X_in,initial_state=state,dtype=tf.float32)
    # print (output[-1])
    # print (final_state)
    output=tf.transpose(output,perm=[1,0,2])
    # output=tf.unstack(tf.transpose(output,perm=[1,0,2]))
    # print (output)
    result=tf.matmul(output[-1,:,:],weights["out"])+biases["out"]
    # result=tf.matmul(final_state[1],weights["out"])+biases["out"]
    return final_state, result

final_state,pred=RNN(x,weights,biases)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    step=0
    while step*batch_size<training_steps:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_inputs,time_steps])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        # print(sess.run(final_state,feed_dict={x:batch_xs}))
        if step%20==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1
        # writer=tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
        # writer.close()





