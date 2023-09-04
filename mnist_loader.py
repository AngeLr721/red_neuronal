#Mnist_loader será la librería que cargue los datos de la biblioteca MNIST a nuestra red para poder entrenarala
#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
#En esta parte estamos definiendo a la función Load_data cuya función será la de cargar los datos de la librería MNIST
    f = gzip.open('mnist.pkl.gz','rb')
    #La primera línea de esta función abre el archivo "minist.pkl" que es un archivo comprimido por lo que se tiene que abrir con la libreria Gzip
    #Este abre dicho archivo en modo binario (rb)
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    #Esta línea carfa los datos con la función "pickle.load" que pertenece a la librería "Pickle" para carfarlos en la variable "f"
    #Los datos descomprimidos del banco de datos MNIST se almacenan en tre variables
    f.close()
    #Esta línea cierra el arcihivo "mnist.pkl" que se había abierto antes
    return (training_data, validation_data, test_data)
    #Esta hace que la funcipon devuleva una datos en un conjuntos de 3 clasificaciones
    #datos de entrenamiento, datos de validación y datos de prueba 
    
def load_data_wrapper():
#Definimos la función 'load_data_wrapper' esta función nos ayduará a preprocesar los datos de la librería MNIST ante de usarlo en nuestro modelos de aprendizaje
    tr_d, va_d, te_d = load_data()
    #Con esta línea llamamos a la función definida anteriormente y cargamos los datos obtenidos en tres conjuntos de datos que corresponden a la salida de la función
    #Asignamos los datos de entreamaniento a 'tr_d', los datos de validación a 'va_d' y el conjunto de pruebas a 'te_d'
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #En esta línea tranformamos la matriz de 28x28 que nos arroja el conjunto de datos de entreamiento ('tr_d[0]) a un vector usando la función 'np.reshape'
    #Este vector 728 entradas se alamcena como uan matriz de 728x1 a la cual nombramos como 'training_inputs'
    training_results = [vectorized_result(y) for y in tr_d[1]]
    #La función 'vectorized_result' toma los datos de entrenamiento, los vectoriza y los guarda en la 'training_results'
    training_data = zip(training_inputs, training_results)
    #Esta función mezcla los datos de las listas 'training_inputs' y 'training-results' y las junta para que sean iteradas solo una vez
    #Al final se almacenan los datos en 'training_data'
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    #Los datos de entrada pasan por el mismo proceso que los datos anteiores, se conviertenen en vectores de 278x1
    validation_data = zip(validation_inputs, va_d[1])
    #En esta líena los datos vectorizados se 'pegan' con las listas anteriores
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    #Ahora pasa lo mismo con los datos de prueba
    test_data = zip(test_inputs, te_d[1])
    #Y vulven a ser 'pegados' en una lista
    return (training_data, validation_data, test_data)
    #Fianlmente se devuelven los datos en un trío de datos procesados
    #'training_data', los datos de entrenamiento, los datos de validación en 'validation_data' y los de prueba en 'test_data'

def vectorized_result(j):
    #La función definida ahora toma un número entero 'j' que representa un dígito de 0 a 9 y devuelve un vector unitario de 10 dimesiones donde j=1 y el resto de valores son 0
    #Cada una de estas salidas está relacionada con una neurona de salidad de la red
    e = np.zeros((10, 1))
    #Crea un vector de 10 filas iniciando con los 0
    e[j] = 1.0
    #Esta parte establece la posición 'j' del vector y cambia el valor de dicha posición por un 1
    return e
    #El vector resultante se devuelve como salida de la función en formato 'one-hot', decir con sólo una de las celdas disntita de cero en el vector resultado 