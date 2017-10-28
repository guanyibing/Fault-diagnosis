import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from ZCSJ import X_train, y_train,X_test,y_test

TIME_STEPS = 20     # same as the height of the image
INPUT_SIZE = 20     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 4
CELL_SIZE = 50
LR = 0.001

X_train = X_train.reshape(-1,20, 20)
X_test = X_test.reshape(-1,20, 20)
y_train = np_utils.to_categorical(y_train, num_classes=4)
y_test = np_utils.to_categorical(y_test, num_classes=4)

model = Sequential()
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
adam = Adam(LR)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

