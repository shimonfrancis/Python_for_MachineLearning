import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/melb_data.csv')
data.head()
y = data.Price
# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'],axis=1)
melb_predictors.head()
x = melb_predictors.select_dtypes(exclude=['object'])
x.head()
# Divide data into training and validation subsets
x_train , x_val , y_train, y_val = train_test_split(x,y,train_size = 0.8,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Function for comparing different approaches
def score_dataset(x_train,x_val,y_train,y_val):
    model = RandomForestRegressor(n_estimators=10,random_state=0)
    model.fit(x_train,y_train)
    predict = model.predict(x_val)
    return mean_absolute_error(y_val,predict)

# Get names of columns with missing values
cols_with_missing = [col for col in x_train.columns if x_train[col].isnull().any()]
redudced_xtrain = x_train.drop(cols_with_missing,axis=1)
reduced_val = x_val.drop(cols_with_missing,axis=1)
print(score_dataset(redudced_xtrain,reduced_val,y_train,y_val))

from sklearn.impute import SimpleImputer
#Imputation
my_imputer = SimpleImputer()
Imputed_xtrain = pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_xval = pd.DataFrame(my_imputer.transform(x_val))
Imputed_xtrain.columns = x_train.columns
imputed_xval.columns = x_val.columns
print(score_dataset(Imputed_xtrain,imputed_xval,y_train,y_val))

# An extension to imputation
# Make copy to avoid changing original data (when imputing)

x_train_plus = x_train.copy()
x_val_plus = x_val.copy()
# Make new columns indicating what will be imputed
for col in cols_with_missing:
    x_train_plus[col+'col was missing']=x_train_plus[col].isnull()
    x_val_plus[col+'col was missing'] = x_val_plus[col].isnull()
# Imputation
my_imputer = SimpleImputer()
imputed_x_train_plus = pd.DataFrame(my_imputer.fit_transform(x_train_plus))
imputed_x_val_plus = pd.DataFrame(my_imputer.transform(x_train_plus))
imputed_x_train_plus.columns = x_train_plus.columns
imputed_x_val_plus.columns = x_val_plus.columns
print(score_dataset(imputed_x_train_plus,imputed_x_val_plus,y_train,y_val))
