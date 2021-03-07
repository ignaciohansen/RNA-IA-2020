# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:45:30 2020

@author: Nacho
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Dataset-Banco.csv')

#X = dataset.iloc[:, [6,5,8,12,10,11,3,9]].values #Datos entrada (Edad,Sexo,Balance,SalarioEstimado,PoseeTarjetaCredito,EsMiembroActivo,Puntaje,Num de productos)
X = dataset.iloc[:, [6,5,8,12,10,11]].values  # Datos entrada (Edad,Sexo,Balance,SalarioEstimado,PoseeTarjetaCredito,EsMiembroActivo)
Y = dataset.iloc[:, [13]].values              # Dato salida (Abandono)

# Codificar datos categóricos - Sexo
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
standarScaler = StandardScaler()
X_train = standarScaler.fit_transform(X_train)
X_test = standarScaler.transform(X_test) 

import keras
from keras.models import Sequential  # Lib para inicializar parametros de la RNA
from keras.layers import Dense # Lib para crear las capas 
from keras.layers import Dropout # Lib para las capas - coeficiente de abandono

# Inicializar la RNA
RNA = Sequential() 

# Añadir la capa de entrada y primera capa oculta
# CAPA ENTRADA
RNA.add(Dense(units = 8, kernel_initializer = "uniform",  activation = "relu", input_dim = 6))
#RNA.add(Dense(units = 8, kernel_initializer = "uniform",  activation = "relu", input_dim = 8))
RNA.add(Dropout(p = 0.1))

# Añadir la capa de salida
RNA.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
RNA.compile(optimizer = "sgd", loss = "mean_squared_error", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
rna_historico = RNA.fit(X_train, Y_train,  batch_size = 15, epochs = 300, validation_data=(X_test, Y_test),) 

Y_pred  = RNA.predict(X_test)
Y_pred = (Y_pred>0.5)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test, Y_pred)

exactitud=(cm[0][0]+cm[1][1])/cm.sum()
print("Exactitud = " + str(exactitud))

precision=cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision = " + str(precision))

recuperacion = cm[0][0]/(cm[0][0]+cm[0][1])
print("Recuperacion = " + str(recuperacion))
      
F1Score = 2 * ((precision * recuperacion / (recuperacion+precision)))
print("F1-Score = " + str(F1Score))


plt.figure(0)   
plt.plot(rna_historico.history['accuracy'],'r')  
plt.plot(rna_historico.history['val_accuracy'],'g')    
plt.rcParams['figure.figsize'] = (12, 6)  
plt.xlabel("Numero de repeticiones")  
plt.ylabel("Exactitud")  
plt.title("Exactitud de entrenamiento vs Exactitud de testeo")  
plt.legend(['Entrenamiento','Test'], loc='upper left')


plt.figure(1)   
plt.plot(rna_historico.history['loss'],'r')  
plt.plot(rna_historico.history['val_loss'],'g')    
plt.rcParams['figure.figsize'] = (12, 6)  
plt.xlabel("Numero de repeticiones")  
plt.ylabel("Porcentaje de Error")  
plt.title("Porcentaje de error en entrenamiento vs Porcentaje de error en testeo")  
plt.legend(['Entrenamiento','Test'], loc='upper left')




