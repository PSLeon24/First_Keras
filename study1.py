from keras.layers.core import Dense
from keras.models import Sequential
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="random_uniform"))
