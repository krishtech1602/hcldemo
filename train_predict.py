import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# load dataset
data = pd.read_csv("traffic_data.csv")

# encode congestion labels
le = LabelEncoder()
data['congestion'] = le.fit_transform(data['congestion'])

# features & target
X = data[['vehicle_count','avg_speed','time_of_day','weather']]
y = data['congestion']

# train model
model = DecisionTreeClassifier()
model.fit(X,y)

# test prediction
sample = [[220, 28, 18, 0]]  # try changing values
prediction = model.predict(sample)

print("Traffic Level:", le.inverse_transform(prediction)[0])