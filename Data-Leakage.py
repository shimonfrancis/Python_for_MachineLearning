import pandas as pd
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/AER_credit_card_data.csv', true_values = ['yes'], false_values = ['no'])
data.head()
y = data.card
x = data.drop(['card'],axis=1)
x.head()
x.shape[0]
print(x.shape[0])
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
vals = cross_val_score(pipeline,x,y,cv=5,scoring='accuracy')
vals.mean()

exp_cardholder = x.expenditure[y]
exp_cardholder
exp_noncardholder = x.expenditure[~y]
exp_noncardholder
print('Fraction of those who did not receive a card and had no expenditures: %.2f' %((exp_noncardholder == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f'%(( exp_cardholder == 0).mean()))
#As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.
# Drop leaky predictors from dataset
leaky = ['expenditure','share','active','majorcards']
X = x.drop(leaky,axis=1)
X
# Evaluate the model with leaky predictors removed
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
valsc = cross_val_score(my_pipeline,X,y,cv=5,scoring='accuracy')
valsc.mean()
