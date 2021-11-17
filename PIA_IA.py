#Reconocimiento de "Dígitos"

import math
def sigmoide(x):
    return 1/(1+math.exp(-x))

def producto_punto(v,w):
    return sum(x*y for x,y in zip(v,w))

def salida_neurona(pesos,entradas):
    #El sesgo ya no se suma aparte, ahora está dentro del producto punto
    return sigmoide(producto_punto(pesos,entradas))

def ffnn(red_neuronal,entrada):
    salidas=[]
    for capa in red_neuronal:
        #el sesgo lo incorporamos al producto punto y le damos un
        #valor de 1 para que tome el peso del sesgo de la neurona
        entrada=entrada+[1]
        #salida_neurona le pasa el producto a la sigmoide
        salida=[salida_neurona(neurona,entrada) for neurona in capa]
        salidas.append(salida)
        #la salida de esta capa es la entrada de la siguiente:
        entrada=salida
    print(salidas[-1])
    return salidas

def backpropagation(xor_nn, v_entrada, v_objetivo):
    salidas_ocultas,salidas=ffnn(xor_nn,v_entrada)

    salida_nuevo=[]
    oculta_nuevo=[]
    alfa=0.1        #usualmente 0.1 es un buen valor

    error=0.5*sum((salida-objetivo)*(salida-objetivo) for salida, objetivo in zip(salidas,v_objetivo))
    salida_deltas=[salida*(1-salida)*(salida-objetivo) for salida, objetivo in zip(salidas,v_objetivo)]
    for i, neurona_salida in enumerate(xor_nn[-1]):
        for j, salida_oculta in enumerate(salidas_ocultas+[1]):
            neurona_salida[j]-=salida_deltas[i]*salida_oculta*alfa
        salida_nuevo.append(neurona_salida)
    ocultas_deltas=[salida_oculta*(1-salida_oculta)*
                    producto_punto(salida_deltas,[n[i] for n in xor_nn[-1]])
                    for i, salida_oculta in enumerate(salidas_ocultas)]
    for i, neurona_oculta in enumerate(xor_nn[0]):
        for j, input in enumerate(v_entrada+[1]):
            neurona_oculta[j]-=ocultas_deltas[i]*input*alfa
        oculta_nuevo.append(neurona_oculta)
    return oculta_nuevo, salida_nuevo, error

#Entradas

entradas=[[1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1], #0
          [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0], #1
          [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1], #2
          [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1], #3
          [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1], #4
          [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1], #5
          [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1], #6
          [1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], #7
          [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1], #8
          [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]] #9

def imprimir(entrada):
    print("")
    print("\t",end='')
    y=''.join([str(x) for x in entrada])
    y=y.replace("0",".")
    y=y.replace("1","@")
    for i,z in enumerate(y):
        print(z,end='')
        if (i+1)%5==0:
            print("")
            print('\t',end='')

for i,v in enumerate(entradas):
    imprimir(v)
print("\n")

objetivos=[[1 if i==j else 0 for i in range(10)] for j in range(10)]

import random
from functools import partial
rand = partial(random.randint)
tamaño_entrada=25 #cada vector entrada tendrá 25 dimensiones
numero_ocultas=5  #número de neuronas en la capa oculta
tamaño_salida=10  #10 neuronas de salida para producir un vector de 10 dimensiones

def pesos():
    #pesos random entre [-1,1]
    random.seed(7)
    #ocupamos 1 peso por cada input más el peso del sesgo
    capa_oculta=[[random.randint(-100,100)/100 for _ in range(tamaño_entrada+1)]
                 for _ in range(numero_ocultas)]
    #ocupamos 1 peso por cada neurona de la capa oculta más el peso #del sesgo
    capa_salida=[[random.randint(-100,100)/100 for _ in range(numero_ocultas+1)]
                 for _ in range(tamaño_salida)]
    return capa_oculta, capa_salida

capa_oculta, capa_salida=pesos()
xor_nn=[capa_oculta,capa_salida]

promedio_errores_cuadrados=1
i=1
while promedio_errores_cuadrados>0.0005:
    #normalizamos los datos de entrada a [-1,1]
    #mediante la fórmula
    #v'=(v-min)/(max-min)*(newmax-newmin)+newmin
    oculta,salida,error1=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1]], objetivos[0])
    xor_nn=[oculta,salida]
    oculta,salida,error2=backpropagation(xor_nn,[x*2-1 for x in [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]], objetivos[1])
    xor_nn=[oculta,salida]
    oculta,salida,error3=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1]], objetivos[2])
    xor_nn=[oculta,salida]
    oculta,salida,error4=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]], objetivos[3])
    xor_nn=[oculta,salida]
    oculta,salida,error5=backpropagation(xor_nn,[x*2-1 for x in [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1]], objetivos[4])
    xor_nn=[oculta,salida]
    oculta,salida,error6=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]], objetivos[5])
    xor_nn=[oculta,salida]
    oculta,salida,error7=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]], objetivos[6])
    xor_nn=[oculta,salida]
    oculta,salida,error8=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]], objetivos[7])
    xor_nn=[oculta,salida]
    oculta,salida,error9=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]], objetivos[8])
    xor_nn=[oculta,salida]
    oculta,salida,error10=backpropagation(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]], objetivos[9])
    xor_nn=[oculta,salida]
    promedio_errores_cuadrados=(error1+error2+error3+error4+error5+error6+error7+error8+error9+error10)/10
    i=i+1


print("__________ffnn(xor_nn, [0])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [1])__________")
ffnn(xor_nn,[x*2-1 for x in [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]])[-1]
print("__________ffnn(xor_nn, [2])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [3])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [4])__________")
ffnn(xor_nn,[x*2-1 for x in [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1]])[-1]
print("__________ffnn(xor_nn, [5])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [6])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [7])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]])[-1]
print("__________ffnn(xor_nn, [8])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]])[-1]
print("__________ffnn(xor_nn, [9])__________")
ffnn(xor_nn,[x*2-1 for x in [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]])[-1]