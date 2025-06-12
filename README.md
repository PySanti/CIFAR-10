
# CIFAR-10

El objetivo de este proyecto será crear una red neuronal capaz de identificar elementos en imágenes. El dataset a utilizar será `keras.datasets.cifar10`. Dicho dataset contiene registros de imágenes clasificadas en 10 categorías que revisaremos en la fase de preprocesamiento.

La arquitectura a utilizar será un `MLP`.

Se utilizarán estrategias avanzadas de `hypertunning`, regularización (`l2` y `dropout`), etc.


## Preprocesamiento

En las primeras observaciones encontramos los siguiente:

```
    Shape del X_train: (50000, 32, 32, 3)
    Shape del Y_train : (50000, 1)
    
    Shape del X_test : (10000, 32, 32, 3)
    Shape del Y_test : (10000, 1)

```

Las imagenes son matrices de `32 x 32 x 3`. En el conjunto de entrenamiento contamos con 50.000 imagenes y sus respectivos targets.

En el conjunto de test contamos con 10.000 imagenes, y tambien, sus respectivos targets.

Cada matriz tiene el siguiente formato:

```
[[[249 248 246]
  [240 240 226]
  [239 241 218]
  ...
  [238 246 223]
  [239 246 230]
  [251 254 252]]

 [[226 229 216]
  [151 156 129]
  [142 150 109]
  ...
  [159 172 129]
  [182 191 163]
  [243 248 239]]

 [[216 223 202]
  [118 128  87]
  [108 119  61]
  ...
  [107 122  64]
  [146 157 118]
  [240 246 230]]

 ...

 [[219 223 198]
  [115 121  82]
  [101 110  60]
  ...
  [  0   0   0]
  [ 57  57  56]
  [237 237 236]]

 [[224 227 205]
  [134 139 107]
  [110 117  75]
  ...
  [ 21  20  20]
  [ 79  79  79]
  [235 235 235]]

 [[244 245 235]
  [224 227 208]
  [216 220 196]
  ...
  [194 192 192]
  [205 205 205]
  [249 249 249]]]

```

La **representacion raw** de cada imagen es una matriz de `32 x 32 x 3` donde cada celda contiene un valor entre 0 y 255 que representa el valor par ese canal (RGB).

En el caso de los targets:

```
Distribucion de targets para Y_train

6    5000
9    5000
4    5000
1    5000
2    5000
7    5000
8    5000
3    5000
5    5000
0    5000
Name: count, dtype: int64

Distribucion de targets para Y_test

3    1000
8    1000
0    1000
6    1000
1    1000
9    1000
5    1000
7    1000
4    1000
2    1000
Name: count, dtype: int64

```

Como podemos ver, el conjunto esta completamente equilibrado.

En cuanto a las imagenes, se ven asi:

![Imagen no encontrada](./images/image_1.png)

Como vemos, son muy pequenias (lo cual tiene sentido por que la resolucion es de `32 x 32`).

Cabe mencionar el mapa de targets:

```
0	airplane
1	automobile
2	bird
3	cat
4	deer
5	dog
6	frog
7	horse
8	ship
9	truck
```

### Aplanamiento de targets

Por alguna razon, al cargar el dataset inicialmente, los targets vienen con el siguiente formato:

```
Y_train : [ [x], [y], [z] ...]
```

Usando el metodo `.flatten()` de numpy, los convertimos al siguiente formato:

```
Y_train : [x, y, z, ...]
```

Codigo:

```
from tensorflow import keras
from sklearn.model_selection import train_test_split

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
print(f"""
Forma inicial de Y_train: {Y_train}
Forma inicial de Y_test: {Y_test}

""")
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print(f"""
Forma final de Y_train: {Y_train}
Forma final de Y_test: {Y_test}
""")

```

Resultado:

```
Forma inicial de Y_train: [[6]
 [9]
 [9]
 ...
 [9]
 [1]
 [1]]
Forma inicial de Y_test: [[3]
 [8]
 [8]
 ...
 [5]
 [1]
 [7]]



Forma final de Y_train: [6 9 9 ... 9 1 1]
Forma final de Y_test: [3 8 8 ... 5 1 7]

```

### Division de conjunto de datos

Se dividio el conjunto de `test` en `val-test`.

Usando el siguiente codigo:

```
from tensorflow import keras
from sklearn.model_selection import train_test_split

# carga del dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

# aplanamiento de targets
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# division del conjunto de datos
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, random_state=42, stratify=Y_test, test_size=.5)


print(f"""

    Shape del X_val : {X_val.shape}
    Shape del Y_val : {Y_val.shape}

    Shape del X_test : {X_test.shape}
    Shape del Y_test : {Y_test.shape}
""")


```

Obtuvimos los siguientes resultados:

```
    Shape del X_val : (5000, 32, 32, 3)
    Shape del Y_val : (5000,)

    Shape del X_test : (5000, 32, 32, 3)
    Shape del Y_test : (5000,)

```

### Normalizacion

La idea de la normalizacion es hacer que todas las caracteristicas del vector de entrada (los pixeles), esten en el mismo rango de valores para no introducir sesgos en el modelo dada la escala de las caracteristicas.

Para ello utilizamos el siguiente codigo:

```
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

```

Como sabemos que todos los pixeles tienen valores entre 0 y 255, al dividir entre 255 nos aseguramos de que todos los pixeles tengan valores entre 0 y 1.

Obtuvimos los siguientes resultados:

```

Valor del registro 100 antes de normalizacion 

[[[213 229 242]
  [211 227 240]
  [211 227 240]
  ...
  [151 174 206]
  [151 174 206]
  [149 172 204]]

 [[214 229 241]
  [212 227 239]
  [212 227 239]
  ...
  [152 175 207]
  [152 175 207]
  [151 174 205]]

 [[216 229 239]
  [214 227 237]
  [213 227 237]
  ...
  [153 176 206]
  [153 176 206]
  [151 174 204]]

 ...

 [[145 159 165]
  [136 148 154]
  [143 152 158]
  ...
  [216 217 206]
  [196 197 191]
  [183 183 182]]

 [[139 153 159]
  [129 142 148]
  [129 139 145]
  ...
  [227 228 219]
  [223 224 219]
  [209 209 209]]

 [[137 152 157]
  [143 155 161]
  [136 145 152]
  ...
  [209 209 203]
  [217 217 213]
  [228 228 226]]]

Valor del registro 100 despues de normalizacion 

[[[0.8352941  0.8980392  0.9490196 ]
  [0.827451   0.8901961  0.9411765 ]
  [0.827451   0.8901961  0.9411765 ]
  ...
  [0.5921569  0.68235296 0.80784315]
  [0.5921569  0.68235296 0.80784315]
  [0.58431375 0.6745098  0.8       ]]

 [[0.8392157  0.8980392  0.94509804]
  [0.83137256 0.8901961  0.9372549 ]
  [0.83137256 0.8901961  0.9372549 ]
  ...
  [0.59607846 0.6862745  0.8117647 ]
  [0.59607846 0.6862745  0.8117647 ]
  [0.5921569  0.68235296 0.8039216 ]]

 [[0.84705883 0.8980392  0.9372549 ]
  [0.8392157  0.8901961  0.92941177]
  [0.8352941  0.8901961  0.92941177]
  ...
  [0.6        0.6901961  0.80784315]
  [0.6        0.6901961  0.80784315]
  [0.5921569  0.68235296 0.8       ]]

 ...

 [[0.5686275  0.62352943 0.64705884]
  [0.53333336 0.5803922  0.6039216 ]
  [0.56078434 0.59607846 0.61960787]
  ...
  [0.84705883 0.8509804  0.80784315]
  [0.76862746 0.77254903 0.7490196 ]
  [0.7176471  0.7176471  0.7137255 ]]

 [[0.54509807 0.6        0.62352943]
  [0.5058824  0.5568628  0.5803922 ]
  [0.5058824  0.54509807 0.5686275 ]
  ...
  [0.8901961  0.89411765 0.85882354]
  [0.8745098  0.8784314  0.85882354]
  [0.81960785 0.81960785 0.81960785]]

 [[0.5372549  0.59607846 0.6156863 ]
  [0.56078434 0.60784316 0.6313726 ]
  [0.53333336 0.5686275  0.59607846]
  ...
  [0.81960785 0.81960785 0.79607844]
  [0.8509804  0.8509804  0.8352941 ]
  [0.89411765 0.89411765 0.8862745 ]]]
```

La eleccion de usar `float32` en lugar de `float64` viene dada por las siguientes razones:

1- float32 ocupa la mitad de la memoria
2- float32 tiene capacidad para 8 digitos de precision, mas que suficiente
3- Librerias como tensorflow estan optimizadas para hacer calculos con float32

## Entrenamiento

## Evaluación

