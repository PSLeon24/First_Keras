# 손글씨 숫자 인식 예제
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)  # 재현을 위한 설정

# Set network and learning
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  # 출력 범주 = 숫자 : 0 ~ 9
OPTIMIZER = SGD()  # SGD
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # 학습 데이터 중에 얼마나 검증 데이터로 할당할지 지정

# 데이터 : 무작위로 섞고, 학습 데이터와 테스트 데이터로 나눔
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train은 60000개의 행으로 구성, 28x28의 값을 가짐 --> 60000x784 형태로 변환
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 정규화
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 범주 벡터를 이진 범주 행렬로 변환
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10개의 출력, 최종 단계는 소프트맥스
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER, metrics=['accuracy'])
