# Introduction to Seaborn
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
data = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/fifa.csv')
data.head()
data.shape[0]
plt.plot(100,100)
sbn.lineplot(data=data)

#LineCharts
music = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/spotify.csv',index_col="Date",parse_dates=True)
music.shape[0]
music.head()
music.tail()
plt.figure(figsize=(16,16))
plt.title("spotify_data")
sbn.lineplot(data=music)
#Plot a subset of the data¶
list(music.columns)
plt.figure(figsize=(14,6))
sbn.lineplot(data=music['Shape of You'])
sbn.lineplot(data=music['Despacito'])

#Bar charts and Heat maps
flight = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/flight_delays.csv',index_col="Month")
flight.head()
#Bar Graph
plt.figure(figsize=(16,16))
sbn.barplot(x=flight.index,y=flight['NK'])
# Heat maps
plt.figure(figsize=(100,100))
Heat_maps = sbn.heatmap(data=flight)
Heat_maps = sbn.heatmap(data=flight,annot=True)
#Scatterplot
insurance = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/insurance.csv')
insurance.head()
scatterplot = sbn.scatterplot(x=insurance['bmi'],y=insurance['charges'])
#Regression line or the line that best fits the data
sbn.regplot(x=insurance['bmi'],y=insurance['charges'])
#More than one variable
sbn.scatterplot(x=insurance['bmi'],y=insurance['charges'],hue =insurance['smoker'])
# To add two regression lines
sbn.lmplot(x="bmi",y="charges",hue="smoker",data=insurance)
#  categorical scatter plot for categorical Variables
sbn.swarmplot(y=insurance['charges'],x=insurance['smoker'])
#Distributions
iris = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/iris.csv',index_col='Id')
iris
iris.shape[0]
#Histogram
plt.figure(figsize=(18,18))
sbn.histplot(iris['Petal Length (cm)'])
# kernel density estimate (KDE) plot. In case you're not familiar with KDE plots, you can think of it as a smoothed histogram.
sbn.kdeplot(iris['Petal Length (cm)'])
# 2D KDE plot
sbn.jointplot(x=iris['Petal Length (cm)'],y=iris['Sepal Width (cm)'],kind="kde")
#Color-coded plots
# Histograms for each species
sbn.histplot(data=iris,x='Petal Length (cm)',hue='Species')
# KDE plots for each specie
sbn.kdeplot(data=iris,x='Petal Length (cm)',hue='Species')
