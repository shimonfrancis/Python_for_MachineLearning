import pandas as  pd
from sklearn.ensemble import RandomForestRegressor
file = '/home/shimonfrancis/Downloads/melb_data.csv'
Data = pd.read_csv(file)
data = Data.dropna(axis=0)
features = ['Lattitude','Longtitude','Bathroom','Rooms','Landsize','Bathroom','YearBuilt']
x = data[features]
y = data.Price
from sklearn.model_selection import train_test_split
trainx,valx,trainy,valy = train_test_split(x,y,random_state=0)
model = RandomForestRegressor(random_state=1)
model.fit(trainx,trainy)
Predict = model.predict(valx)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(valy,Predict)
mae
