"""
    This program is an example of a feedforward
    neural network which utilizes TensorFlow. I have
    used a sequential neural network for this model.
    The Pima Indian diabetes data being inputted into
    the model is quantitative and continuous. 60-20-20
    was the split between the training, validation,
    and testing datasets respectively.
    
    To run on local computer you might need to download
    sklearn, tensorflow, and other libraries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow_hub as hub
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("diabetes.csv")

df.head()

df.columns

len(df[df["Outcome"]==1]), len(df[df["Outcome"]==0])

for i in range(len(df.columns[:-1])):
  label = df.columns[i]
  plt.hist(df[df["Outcome"]==1][label], color="blue", label="Diabetes", alpha=0.7, density="True", bins=15)
  plt.hist(df[df["Outcome"]==0][label], color="red", label="No diabetes", alpha=0.7, density="True", bins=15)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

#splitting input (X) and target values (y)
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

#scales input values for women w/ diabetes and women w/o diabetes to be equal
scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

len(transformed_df[transformed_df['Outcome']==1]), len(transformed_df[transformed_df['Outcome']==0])

over = RandomOverSampler()
X, y = over.fit_resample(X,y)
data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

#splitting data into 60-20-20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

#our model is a simple sequential neural net with 16 neurons in the first two
#layers and 1 neuron in the final layer
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(16, activation='relu'),
                             tf.keras.layers.Dense(16, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')
])

#loss function is Binary Cross-Entropy since we have a binary output
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.evaluate(X_train, y_train)

model.evaluate(X_valid, y_valid)

model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)
