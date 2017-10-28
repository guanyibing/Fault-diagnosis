# 序贯模型没法截取一部分构成自动编码器的编码过程，这只是一个全连接【200，100，50，20，4】的序贯模型
from keras.layers import Input, Dense,Activation,Dropout
from keras.models import Sequential
import numpy as np
from keras.regularizers import l1
from zhouchengshuju import train_set,train_label,test_set,test_label

model = Sequential()  #序贯式模型
model.add(Dense(100, activation='relu', input_shape=(200,)))
model.add(Dropout(0.05)) #防止过拟合，每次更新参数时随机断开一定百分比（rate）的输入神经元
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(4, activation='softmax'))
model.summary()
model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_set,train_label,nb_epoch=100,batch_size=100,shuffle=True)
score = model.evaluate(test_set, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.0204376992621
# Test accuracy: 0.997
