import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
melb = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
x = melb[cols_to_use]
x.head()
y = melb.Price
y.head()
pipe = Pipeline(steps=[('impute',SimpleImputer()),('model',RandomForestRegressor(n_estimators=100,random_state=0))])
from sklearn.model_selection import cross_val_score
#The scoring parameter chooses a measure of model quality to report: in this case, we chose negative mean absolute error (MAE).
scores = -1*cross_val_score(pipe,x,y, scoring='neg_mean_absolute_error')
print(scores)
print(scores.mean())
