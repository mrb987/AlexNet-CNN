from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import layers, optimizers, losses
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import numpy as np

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

print(len(train_labels))
print(train_labels)
print(len(test_labels))
print(test_labels)

X_train = train_images.astype('float32') / 255
X_test = test_images.astype('float32') / 255
print(X_train.shape)
print(X_test.shape)

Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)
# --------------------------
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
plt.subplot(2, 1,1)
digit = train_images[24]
plt.imshow(digit, cmap=plt.cm.gray)
plt.subplot(2, 1,2)
digit = X_train[24]
plt.imshow(digit, cmap=plt.cm.gray)
# ------------------------------
input_img = layers.Input(shape=(32, 32, 3))
anet = layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='same')(input_img)
anet = layers.BatchNormalization()(anet)
anet = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(anet)
anet = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(anet)
anet = layers.BatchNormalization()(anet)
anet = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(anet)
anet = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(anet)
anet = layers.BatchNormalization()(anet)
anet = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(anet)
anet = layers.BatchNormalization()(anet)
anet = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(anet)
anet = layers.BatchNormalization()(anet)
anet = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(anet)
anet = layers.Flatten()(anet)
anet = layers.Dense(4096, activation='relu')(anet)
anet = layers.Dropout(0.5)(anet)
anet = layers.Dense(4096, activation='relu')(anet)
anet = layers.Dropout(0.5)(anet)
out_class = layers.Dense(10, activation='softmax')(anet)
cifar_model = Model(input_img, out_class)
# -----------------------------------
cifar_model.compile(optimizer = optimizers.Adam(), loss=losses.binary_crossentropy)
cm = cifar_model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_test, Y_test))
res = cifar_model.predict(X_test)
print(cifar_model.summary())
print(res.shape)
print(res[1,:])
plot_model(cifar_model,to_file='x.png')
