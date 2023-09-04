import mnist_loader
#Importamos las bibliotecas que hemos creado para cargar los datos de MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Se llama a la función de procesamiento que procesa los datos y los regresa en tres conjuntos de datos
training_data = list(training_data)
#Convierte los datos en una lista que será el argumento para la función 'network'

import network
#Ahora se importa la biblioteca donde está cotenida la red neuronal
net = network.Network([784, 30, 10])
#Creamos ahora red neuronal con 784 neuronas de entrada, es decir del largo de nuestras entradas vectorizadas
#30 neuronas intermedias y 10 neuronas, una por cada dígito
net.SGD(training_data, 10, 10, 1, test_data=test_data)
"""Inicia el proceso de entrenamiento de la red utilizando el algorimo SDG
Los argumentos son
    *'Training_Data'
        Datos de entrenamiento
    *'{},{},{}'
        El número de épocas de entrenamiento, el tamaño del lote por cada iterazión del algoritmo SDG y la taza de aprendizaje
    *'test_data=test_data'
        Los datos de prueba se pasan como argumento para evaluar el rendimiento de la red despúes de cada época"""   