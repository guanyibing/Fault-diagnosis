#堆栈自动编码器
from keras.layers import Input, Dense
from keras.models import Model
from zhouchengshuju import train_set,train_label,test_set,test_label

encoding_dim = 20
input_data = Input(shape=(200,))
encoded = Dense(100, activation='relu')(input_data)
encoded = Dense(50, activation='relu')(encoded)
encoder_output = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoder_output)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(200, activation='sigmoid')(decoded)
autoencoder = Model(input=input_data, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(train_set,train_set,nb_epoch=100,batch_size=100,shuffle=True,validation_data=(test_set, test_set))

output=Dense(4,activation='softmax')(encoder_output)
sae=Model(input=input_data,output=output)
sae.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
sae.fit(train_set,train_label,nb_epoch=100,batch_size=100,shuffle=True,validation_data=(test_set, test_label))

print(sae.predict(test_set))
loss_and_metrics = sae.evaluate(test_set, test_label, verbose=0)
print("loss_and_metrics:",loss_and_metrics)
# loss_and_metrics: [0.05289214497525245, 0.98050000000000004]














