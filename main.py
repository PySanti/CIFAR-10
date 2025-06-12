from tensorflow import keras
from sklearn.model_selection import train_test_split

# carga del dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

# aplanamiento de targets
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# division del conjunto de datos
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, random_state=42, stratify=Y_test, test_size=.5)

# normalizacion
print("Valor del registro 100 antes de normalizacion ")
print(X_train[100])

X_train = X_train.astype("float32") / 255.0

print("Valor del registro 100 despues de normalizacion ")
print(X_train[100])

X_test = X_test.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
