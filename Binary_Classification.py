import pandas as pd
from sklearn.model_selection import train_test_split as ts
ion = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/ion.csv')
ion.head()
df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})
ion.shape
x = df.drop('Class',axis=1)
x.shape
y = df['Class']
y.shape
trainx,valx,trainy,valy = ts(x,y,random_state=0)
trainx.shape
trainy.shape
from tensorflow import keras
from tensorflow.keras import layers, callbacks
ann = keras.Sequential([layers.Dense(units=4,activation='relu',input_shape=[35]),layers.Dense(units=4,activation='relu'),layers.Dense(1,activation='sigmoid')])
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
early_stopping = callbacks.EarlyStopping(patience=10,min_delta=0.001,restore_best_weights=True)
history = ann.fit(trainx,trainy,validation_data=(valx,valy),batch_size=512,epochs=1000,callbacks=[early_stopping],verbose=0)
losss = pd.DataFrame(history.history)
losss.loc[:,['loss','val_loss']].plot()
losss.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
losss['val_loss'].min()
losss['val_binary_accuracy'].max()
