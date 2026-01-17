import pandas as pd
from prophet import Prophet

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Get only timestamp and time series data from the dataframe.
data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y'] # Renaming the columns per Prophet's requirements.

model = Prophet()
modelFit = model.fit(data_p)

# Get future data to make predictions.
data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
data_f = data_test[['Timestamp']]
data_f.columns = ['ds']

pred = modelFit.predict(data_f)