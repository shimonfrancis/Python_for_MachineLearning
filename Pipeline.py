import pandas as pd
from sklearn.model_selection import train_test_split
melb = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/melb_data.csv')
melb.head()
y = melb.Price
y.head()
x = melb.drop('Price',axis=1)
x.head()
t_x,v_x,t_y,v_y = train_test_split(x,y,random_state=0)
# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols =[cname for cname in t_x.columns if t_x[cname].nunique()<10 and t_x[cname].dtype=='object' ]
categorical_cols
numerical_cols = [cname for cname in t_x.columns if t_x[cname].dtype in ['int64','float64']]
numerical_cols
cols = categorical_cols + numerical_cols
cols
x_trains = t_x[cols].copy()
x_trains.head()
x_vals = v_x[cols].copy()
x_vals
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')
numerical_transformer
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown ='ignore'))])
categorical_transformer
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num',numerical_transformer,numerical_cols),('cat',categorical_transformer,categorical_cols)])
preprocessor
#Define machine learning model_last
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=0)
