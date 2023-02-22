import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
#The easiest way to create a model in Keras is through keras.Sequential,
#which creates a neural network as a stack of layers.
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/red-wine.csv')
x= data.drop('quality',axis=1)
x
y= data['quality']
y
xtrain,xval,ytrain,yval=train_test_split(x,y,random_state=0)
early_stopping = callbacks.EarlyStopping(min_delta=0.001,patience=20,restore_best_weights=True)
model = keras.Sequential([layers.Dense(units=512,activation='relu',input_shape=[11]),layers.Dense(units=512,activation='relu'),layers.Dense(units=512,activation='relu'),layers.Dense(units=1)])
model.compile(optimizer='adam',loss='mae',)
model.weights
fit = model.fit(xtrain,ytrain,validation_data=(xval,yval),batch_size=256,epochs=500,callbacks=[early_stopping])
history = pd.DataFrame(fit.history)
history.loc[:,['loss','val_loss']].plot()
history['val_loss'].min()
