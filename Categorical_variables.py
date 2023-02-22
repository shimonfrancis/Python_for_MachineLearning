import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/melb_data.csv')
data.head()
y = data.Price
x = data.drop(['Price'],axis=1)
x
x_train,x_val,y_train,y_val = train_test_split(x,y, train_size=0.8, test_size=0.2,random_state=1)
columnsmissing = [col for col in x_train if x_train[col].isnull().any()]
x_train1 = x_train.drop(columnsmissing,axis=1)
x_val1 = x_val.drop(columnsmissing,axis=1)
# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
cardinality_cols = [cname for cname in x_train1.columns if x_train1[cname].nunique()<10 and x_train1[cname].dtype=="object"]
cardinality_cols
# Select numerical columns
numerical_cols = [cname for cname in x_train1.columns if x_train1[cname].dtype in ['int64', 'float64']]
numerical_cols
# Keep selected columns only
my_cols = cardinality_cols + numerical_cols
x_train2 = x_train1[my_cols].copy()
x_val2 = x_val1[my_cols].copy()
x_train2.head()
# Get list of categorical variables
s = (x_train2.dtypes=='object')
object_cols = list(s[s].index)
object_cols
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def scores(x_train2,x_valid2,y_train,y_valid):
    model = RandomForestRegressor(n_estimators=100,random_state=0)
    model.fit(x_train2,y_train)
    prediction = model.predict(x_valid2)
    return mean_absolute_error(y_val,prediction)
# Drop categorical Variables
drop_xtrain = x_train2.select_dtypes(exclude=['object'])
drop_xval = x_val2.select_dtypes(exclude=['object'])
print(scores(drop_xtrain,drop_xval,y_train,y_val))

#Original Encoder
from sklearn.preprocessing import OrdinalEncoder
#copy the values
x_ordinal_train = x_train2.copy()
x_ordinal_val = x_val2.copy()
# Apply ordinal encoder to each column with categorical data
my = OrdinalEncoder()
x_ordinal_train[object_cols] = my.fit_transform(x_train2[object_cols])
x_ordinal_val[object_cols] = my.transform(x_val2[object_cols])
print(scores(x_ordinal_train,x_ordinal_val,y_train,y_val))

#One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
#We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
#setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
hot = OneHotEncoder(handle_unknown='ignore',sparse=False)
x_hot_train = pd.DataFrame(hot.fit_transform(x_train2[object_cols]))
x_hot_val =pd.DataFrame(hot.transform(x_val2[object_cols]))
# One-hot encoding removed index; put it back
x_hot_train.index = x_train2.index
x_hot_val.index = x_val2.index
# Remove categorical columns (will replace with one-hot encoding)
num_x_train = x_train2.drop(object_cols,axis=1)
num_x_val = x_val2.drop(object_cols,axis=1)
# Add one-hot encoded columns to numerical features
one_x_train = pd.concat([num_x_train,x_hot_train],axis=1)
one_x_val = pd.concat([num_x_val,x_hot_val],axis=1)
print(scores(one_x_train,one_x_val,y_train,y_val))
