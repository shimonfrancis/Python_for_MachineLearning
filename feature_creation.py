import pandas as pd
import seaborn as sns
import numpy as np
concrete = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/concrete.csv')
autos = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/autos.csv')
accidents = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/accidents.csv')
customer = pd.read_csv('/home/shimonfrancis/Documents/Data/Ml/customer.csv')
autos["stroke_ratio"] = autos.stroke / autos.bore
autos[["stroke","bore","stroke_ratio"]].head()
autos["displacement"]= (np.pi*((0.05*autos.bore)**2)*autos.stroke*autos.num_of_cylinders)
autos.displacement
# Data visualization can suggest transformations, often a "reshaping" of a feature through powers or logarithms. The distribution of WindSpeed in US Accidents is highly skewed, for instance. In this case the logarithm is effective at normalizing it:
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
sns.kdeplot(accidents.WindSpeed)
sns.kdeplot(accidents.LogWindSpeed)
#In Traffic Accidents are several features indicating whether some roadway object was near the accident. This will create a count of the total number of roadway features nearby using the sum method:# In Traffic Accidents are several features indicating whether some roadway object was near the accident. This will create a count of the total number of roadway features nearby using the sum method:
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay","Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop","TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
accidents[roadway_features + ["RoadwayFeatures"]].head(10)

#You could also use a dataframe's built-in methods to create boolean values
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water","Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)
concrete[components + ["Components"]].head(10)
customer[["Type", "Level"]] = (customer["Policy"].str .split(" ", expand=True))  # Create two new features

        # from the Policy feature
                           # through the string accessor
         # by splitting on " "
                                 # and expanding the result into separate columns

customer[["Policy", "Type", "Level"]].head(10)
#You could also join simple features into a composed feature if you had reason to believe there was some interaction in the combination:
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()
# we have Group transforms, which aggregate information across multiple rows grouped by some category
customer["AverageIncome"] = (customer.groupby("State") ["Income"].transform("mean"))
customer[["State", "Income", "AverageIncome"]].head(10)
customer.head()
customer["state frequency"] = customer.groupby("State")["State"].transform("count")/customer.State.count()
customer[["State","state frequency"]]
customer["Jack"] = customer.groupby("State")["CustomerLifetimeValue"].transform("count")/customer.State.count()
customer[["State","Jack"]]
