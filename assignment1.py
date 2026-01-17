from pygam import LinearGAM, s, f
import pandas as pd

# Get data.
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Grab timestamp columns and data.
x = data[['year', 'month', 'day', 'hour']]
y = data['trips']

# Create the model.
model = LinearGAM(f(0) + f(1) + s(2) + s(3))
# Fit the model.
modelFit = model.fit(x, y)

# Make predictions using the fitted model.
pred = modelFit.predict(x)
pred = pd.DataFrame({'trend': pred}) # Convert to DF.
pred = pred.tail(744)
pred = pred.reset_index(drop=True)