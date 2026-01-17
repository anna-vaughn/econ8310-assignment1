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

# Create an empty dataframe with dates for future periods
future = modelFit.make_future_dataframe(periods=744, freq='h')
# Fill in dataframe wtih forecasts of `y` for the future periods
pred = modelFit.predict(future)