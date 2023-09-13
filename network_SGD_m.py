import random
import math

# Third-party libraries
import numpy as np

class Network(object):
#Definimos la clase 'Network', es decir de una red neuronal
    def __init__(self, sizes):
    #En esta parte el constructor '__init__' inicializa la red neuronal, es decir crea los atributos que debe de tener la clase, en este caso la red neuronal
    #Por otro lado especificamos la arquitectura de la red con la función 'sizes' que nos indica el número de capas que tendrá nuestra red neuronal
        self.num_layers = len(sizes)
        #Esta linea calcula y almacenza el número de capas de la red neuronal, como 'sizes' contiene el numero de neuronas de la red esta lista tedrá el largo que la red de capas
        self.sizes = sizes
        # Esta línea nos dará información de la arquietectura de la red en toda instancia de la clase 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #Esta es la lista de sesgos de la red para las capas ocultas de la red
        #Usamos la funcion de Numpy 'np.random.randn(y,1)' para generar un vector de largo 'y' en donde se alamacernarán los sesgos de esas mismas 'y' neuronas
        #Los valores de estos sesgos se inicializarán aleatoriamente uzando una gaussiana con media 0 y varianza 1
        #Estas listas de pesos se crean para todas las capas exceto para la capa de entrada
        self.weights = [np.random.randn(y, x)
        #Es esta línea se crea un matriz de dimenciones x*y donde almacenaremos los pesos para cada conexión en la red. Se utiliza el metodo que en la línea anteior                
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #El número de conexiones entre una capa y la otra se determinan en esta línea de código
        #Este depende del tamaño de las capas 'sizes' y luego se crean pares con lo tamaños de las capa consecutivas 
        
    def feedforward(self, a):
    #Definimos la función feedforward, esta función dentro de la clase 'network', esta representa el proceso de de propagación de la red
    #Dado un conjunto de entradas 'a' la función calcula y devuelve la salidad de la red despues de aplicar las operaciones de activación en cada capa de la red 
    #Esta función toma como argumento un vector 'a' que representa la entrada que se propaga por el red
        for b, w in zip(self.biases, self.weights):
        #Este bucle itera sobre la lista de sesgos y las matrices pesos de la red neuronal, para hacer sólo un ciclo de iteraciones se usa la función 'zip' que 'pega' las listas en una
        #Para cada capa de la red se toma un sesgo 'b' y una matriz de pesos 'w'
            a = sigmoid(np.dot(w, a)+b)
        #En cada iteración del bucle se realizan la siguiente operación 
            #'np.dot(w,a)'Primero se calcula prodcuto punto entre la matriz de pesos 'w' de la capa y el vector de activación 'a' de la capa anterior
            #'...+b' Al producto anterior se le suman los sesgos 'b' del vector de sesgos
            #'sigmoid(...)' finalemente se aplica la función de activación que en nuestro ejemplo es la función sigmoide la cual toma una entrada real y la normaliza para representar la activación de la neurona
        return a
        #Al final se devuelve 'a' que representa la salida de la red depues de ejecutar la rpopagación hacia adelante

    def SGD_m(self, training_data, epochs, mini_batch_size, eta, momentum,
            test_data=None):
        #Ahora definimos la función que nos representará el algoritmo 'Stochastic Gradient Descent', este algortimo se encargará de entrenar las redes neuronales mediante el metodo SGD
        #Definimos tambien los argumentos que tendrá la función con los datos de entramiento 'training_data', las epocas de entrenamiento 'epochs', los paquetes de datos 'mini_batch_size' y la taza de crecimeinto 'eta'
        #y los datos de prueba 'test_data' los cuales opcionales y se pueden utilizar para evaluar el rendimiento de la red en cada época

        #Añadimos el parametro de 'momentum' para poder entrenar de manera más eficienta a la red neuronal
        
        training_data = list(training_data)
        #Convertimos los datos de entrenamiento en una lista 'training_data' de duplas (x,y) donde 'x' es la entrada de entramiento y 'y' es el dato de contraste con la salida
        n = len(training_data)
        #Se calcula la longitud total de los datos de entrenamiento
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        #Verificamos si se proporcionan los datos de prueba, se convienten en una lista y se calculan la longitud de los datos para evaluar posteriormente el rendimiento de la red
             
        velocidad_w=[np.zeros(w.shape) for w in self.weights]
        velocidad_b=[np.zeros(b.shape) for b in self.biases]
        #Definimos las funciones de velocidad de cambio de las variables w y b que se usarán el el algotirmo 'SGD+momentum'
        
        for j in range(epochs):
        #Iniciamos ahora un blucle iterado sonbre el número especificado de épocas de entrenamiento
            random.shuffle(training_data)
            #'Barajamos' los datos de entremiento para evitar que la red se ajuste demasaido a un orden específico de ejemplos
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            #Ahora los datos de entrenamiento se dividen en paquetes o lotes 'mini_batches'de acuerdo a un tamaño especificado en los argumentos de la función 'mini_batches_sizes'
            #Esto nos crea una lista de lotes donde cada lote es un lista de ejemplos
            for mini_batch in mini_batches:
                self.update_mini_batch_momentum(mini_batch, eta, momentum, velocidad_w, velocidad_b)
            #Se itera a través de cada lote pequeño y depués de acabar cada lote se actualizan los pesos y los sesgos de la red neuronal en función de ese lote (función 'update_mini_batch'), Esta parte es fundamental para el entrenamiento
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
            #Despues de cada época, se evalúa el rendimiento de la red neuronal utilizando la función 'evaluate' y se imprime el progreso del entrenamiento en términos de la precisión de la red con los datos de prueba
            
    def update_mini_batch_momentum(self, mini_batch, eta, momentum, velocidad_w, velocidad_b):
        """Definimos la función antes declarada que se encargará de actualizar los pesos y los sesgos de la red neuranal con el algoritmo SGD aplicado a los mini batches
        Definimos sus argumentos:
            'mini_batch'
            La lista de duplas (x,y) donde (x) es la entrada de entrenamiento e 'y' la salida correspondiente
            'eta'
            La taza de aprendizaje que controla la magnitud de las actualizaciones de los pesos durante el entrenamiento"""
            #El nuevo argumento de 'momentum' nos permite establecer un parámetro para el momentum, es siempre al rededor de 0.9
            
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Se inician las listas 'nabla_b' y 'nabla_w' con ceros, estas listas guardaran las los gradientes de los sesgos y pesos
        for x, y in mini_batch:
        #Se inicia el bucle iterado para cada dupla (x,y)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Para cada ejemplo se llama la función 'backprop' con la que se calculan los gradientes de 'b' y 'w'
            #Estos gradientes representan como deben de cambiar los sesgos y pesos para reducir el error de predicción
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #Los gradientes calculados se agregan a las acumulaciones 'nabla_b' y 'nabla_w', esta operación se itera para cada par de datos y se promedian
        velocidad_w = [momentum*vw - (eta/len(mini_batch))*nw for vw, nw in zip(velocidad_w, nabla_w)]
        velocidad_b = [momentum*vb - (eta/len(mini_batch))*nb for vb, nb in zip(velocidad_b, nabla_b)]
        
        self.weights = [w + vw for w, vw in zip(self.weights, velocidad_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, velocidad_b)]
        #Finalmente los pesos y los sesgos de la red se actualizan usando los gradientes promediados usando la actualización de SGD donde 'eta' es la taza de aprendizaje y se dividen por el tamaño del lote para promediar todo los gradientes acumulados
    
    def backprop(self, x, y):
    #Definimos la función 'backprop', usamos para calcular los gradientes de los sesgos y los pesos de la red neuronal
    #Definimos sus argumentos (x,y) donde 'x' es la entrada de entrenamiento y 'y' la salida deseada en 'x'
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Se inician las listas con ceros, estás se usarán para almacenar los gradientes mientras se realizan los cálculos durante el algoritmo 'backpropagation' 
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        """"En la primera parte de este trozo de código se inicia el algoritmo 'backpropagation' realizando una propagación hacia adelante 'feedforward' en la red neuronal, Se inicia la activación 'activation' con las entradas 'x'
        Se almacenan las activaciones en 'activactions' y en 'zs' y los vectores 'z' en cada capa de la red
        
        El bucle 'for' itera sobre las listas de sesgos y pesos de la red neuronal
        En cada iteración del bucle se calcula la suma en 'z' para una capa mediante 'np.dot(w,activation)+b' donde 'activation' es la capa de activación de la capa anterior , las salidas de 'z' se guardan en la 'zs' como se había mencionado
        Al final se calcula la activación de la capa ctual usando la función de activación 'sigmoid(z)', y se agreaga a la lista 'activations'
        """
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """Despues del feedfodward se inicia el proceso de 'backpropagation'. Se calcula la función de costo con respecto a la activación de la última capa y se multiplica por la derivada de la función de activación aplicada en 'z'
        El resultado se almacena en la lista 'nabla_b' que corresponde a la última capa y se calcula el gradiente con respecto a los pesos de la ultima capa y se alamcenana en 'nalbla_w[-1]
        Python no tiene problemas que usemos indices negativos
        """
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    """Ahora se realiza un bucle desde la penultima capa hasta la segunda capa, en cada iteración del bucle se calcula 'delta' para la capa anterior utilizando el algoritmo de 'retropropagación'
    Se calcula 'z' para la capa actual, se calcula a derivada de la función de activación para 'z' y se actualiza 'delta' multiplicando por la transpuesta de la matriz de pesos de la capa siguiente y se almacenan los gradientes en las listas 'nabla' correspondientes
    Finalmente la función 'backprop¿ devuelve la dupla (nabla_b, nabla_w) con los gradientes de los sesgos y los pesos para la red neuronal"""
    
    def evaluate(self, test_data):
    #Devuelve el número de entradas de prueba para los cuales la red neuronal genró resultados correctos   
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    """La función calcula 'test_results'. Para cada par (x,y) en el conjunto de datos de prueba
        *'self.feedforward(x)'
            Se pasa la entrada 'x' a través de la red neuronal con la función 'feedforward'
        *'np,argmax(...)'
            Encuentra el índice de la neurona en la capa de salida con la activación más alta, esto represnta la predicción de la red
        *'(np.argmax(self.feedforward(x)), y)'
            Se crea una pareja de datos que contiene la predicción de la red y la etiqueta real 'y' del conjunto de datos de prueba
    Finalmente la función 'evaluate' devuelve el número de entradas de prueba para las cuales la red acertó en su predicción, esto se logra cuando coincide con la etiqueta ral 'int(x==y)'"""
        
    def cost_derivative(self, output_activations, y):
    #Esta función nos devuelve el vector de derivadas parciales para las activaciones de salidas
    #Esta función tiene como argumentos la función de activación de salida y las etiquetas reales
        return (output_activations-y)
        #Devuelve la diferencia entre las activaciones de salida de la red y las etiquetas reales
        #Este resultado representa el gradiente de la función de costo con respectoa a las activaciones de salida
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
#Se define la función sigmoide que usamos como función de activación
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#Esta es la derivada de la función de activación