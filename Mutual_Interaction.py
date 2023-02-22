import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/autos.csv')
data.head()
x = data.copy()
y = x.pop('price')
x.head()
y.head()
# Label encoding for categoricals
for colname in x.select_dtypes("object"):
    x[colname], _ = x[colname].factorize()
# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = x.dtypes == int
from sklearn.feature_selection import mutual_info_regression
def make_mi(x,y,discrete_features):
    mi_scores = mutual_info_regression(x,y,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores,index=x.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_scores = make_mi(x,y,discrete_features)
mi_scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
plot_mi_scores(mi_scores)
sns.regplot(x="curb_weight", y="price", data=data)
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=data);
