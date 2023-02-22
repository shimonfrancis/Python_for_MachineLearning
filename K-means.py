import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/housing.csv')
data.head()
x = data.loc[:, ["median_income", "latitude", "longitude"]]
x.head()
#Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values. Our features are already roughly on the same scale, so we'll leave them as-is.
kmeans = KMeans(n_clusters=6)
x["cluster"]= kmeans.fit_predict(x)
x["cluster"] = x["cluster"].astype("category")
x.head()
sns.relplot(x="longitude",y="latitude",hue="cluster",data=x)
x["median_house_value"]=data["median_house_value"]
x.head()
sns.catplot(x="median_house_value",y="cluster",data=x)
