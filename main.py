from tensorflow import keras
import pandas
from utils.show_image import show_image

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()


show_image(X_test[500])
