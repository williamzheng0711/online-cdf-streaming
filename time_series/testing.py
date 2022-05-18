import pandas as pd
from darts import TimeSeries
import numpy as np
import torch

# Read a pandas DataFrame
df = pd.read_csv('AirPassengers.csv', delimiter=",")

# Create a TimeSeries, specifying the time and value columns
# series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')


series = TimeSeries.from_values( np.array( np.arange(0, 600, step = 1)) )
print(series)
# Set aside the last 36 months as a validation series
train, val = series[:-36], series[-36:]

from darts.models import ExponentialSmoothing
from darts.models import ARIMA
from darts.models import RNNModel

model = ExponentialSmoothing()
model2 = ARIMA(p = 4, d = 2, q= 1)
# model3 = RNNModel(
#     model="LSTM",
#     hidden_dim=20,
#     dropout=0,
#     batch_size=16,
#     n_epochs=300,
#     optimizer_kwargs={"lr": 1e-3},
#     model_name="Air_RNN",
#     log_tensorboard=True,
#     random_state=42,
#     training_length=20,
#     input_chunk_length=14,
#     force_reset=True,
#     save_checkpoints=True,
# )

model2.fit(train)
prediction = model2.predict(len(val), num_samples=10)

import matplotlib.pyplot as plt

series.plot()
prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()