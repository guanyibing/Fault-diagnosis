import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam
from ZCSJ import X_train, y_train,X_test,y_test

# X shape (36,000 20x20), y shape (36,000, )
X_train = X_train.reshape(-1,20, 20,1)
X_test = X_test.reshape(-1,20, 20,1)
y_train = np_utils.to_categorical(y_train, num_classes=4)
y_test = np_utils.to_categorical(y_test, num_classes=4)

model = Sequential()
model.add(Convolution2D(input_shape=(20,20,1),filters=32,kernel_size=5,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first'))
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# Fully connected layer 1 input shape (64 * 5 * 5) = (1600), output shape (1024)
model.add(Flatten()) # 将响应转换为一维向量1600
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.1)) # Dropout
model.add(Dense(4))
model.add(Activation('softmax'))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
print('Training ------------')
model.fit(X_train, y_train, epochs=1, batch_size=100)
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
# without dropout,test accuracy:  0.92275
# dropout:0.1,test accuracy:0.9725,0.9635
# dropout:0.3,test accuracy:0.911
# dropout:0.15,test accuracy:0.93075