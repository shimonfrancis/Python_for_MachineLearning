import pandas as pd
from sklearn.tree import DecisionTreeRegressor
file = '/home/shimonfrancis/Downloads/melb_data.csv'
Data = pd.read_csv(file)
pure = Data.dropna(axis=0)
Data

J = ['Lattitude','Longtitude','Bathroom','Rooms']
x = Data[J]
y= Data.Price
# Define Model
model = DecisionTreeRegressor(random_state=1)
# Fit Model
model.fit(x,y)
# Predict Model
Predict = model.predict(x)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,model.predict(x))
# Evaluate model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
trainx,valx,trainy,valy = train_test_split(x,y,random_state=0)
model_train = DecisionTreeRegressor(random_state=1)
# we are fitting the model with training data
model_train.fit(trainx,trainy)
# We are predicting the values with value x on trained model
train_predicted = model_train.predict(valx)
mean_absolute_error(valy,train_predicted)
# underfitting and overfitting
def get_mae(max_leaf_nodes,trainx,valx,trainy,valy):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1)
    model.fit(trainx,trainy)
    pred = model.predict(valx)
    mae = mean_absolute_error(valy,pred)
    return(mae)
for max_leaf_nodes in [5,50,500,5000]:
    node_mae = get_mae(max_leaf_nodes,trainx,valx,trainy,valy)
    print(max_leaf_nodes,node_mae)
#final model with best tree
model_last = DecisionTreeRegressor(max_leaf_nodes = 500,random_state=1)
model_last.fit(x,y)
model_last.predict(x)
