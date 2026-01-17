import pandas as pd
from prophet import Prophet

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Get only timestamp and time series data from the dataframe.
data_p = data[['Timestamp', 'trips']]
data_p.columns = ['ds', 'y'] # Renaming the columns per Prophet's requirements.

model = Prophet()
model = model.fit(data_p)

# Create an empty dataframe with dates for future periods
future = model.make_future_dataframe(periods=744, freq='h')
# Fill in dataframe wtih forecasts of `y` for the future periods
modelFit = model.predict(future)

# Get only forecasted variables.
pred = modelFit.loc[(modelFit['ds'] >= '2019-01-01 00:00:00')]
pred = pred[['ds', 'trend']]
pred = pred.reset_index(drop=True)