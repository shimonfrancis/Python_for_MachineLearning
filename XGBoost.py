import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
x = data[cols_to_use]
y = data.Price
xtrain,xval,ytrain,yval = train_test_split(x,y,random_state=0)
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=4)
model.fit(xtrain,ytrain,early_stopping_rounds=5,eval_set=[(xval,yval)],verbose=False)
predict = model.predict(xval)
mae = mean_absolute_error(yval,predict)
print(mae)
