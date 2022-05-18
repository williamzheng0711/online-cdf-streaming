import pandas as pd
from darts import TimeSeries
import numpy as np

# Read a pandas DataFrame
df = pd.read_csv('AirPassengers.csv', delimiter=",")

# Create a TimeSeries, specifying the time and value columns
# series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')


series =  np.arange(0, 600, step = 2)
# Set aside the last 36 months as a validation series
train, val = series[:-36], series[-36:]

from darts.models import ExponentialSmoothing
from darts.models import ARIMA

model = ExponentialSmoothing()
model2 = ARIMA()
model2.fit(train)
prediction = model2.predict(len(val), num_samples=1000)

import matplotlib.pyplot as plt

series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()