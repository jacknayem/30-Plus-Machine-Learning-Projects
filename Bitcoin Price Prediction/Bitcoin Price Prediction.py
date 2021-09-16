#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from prophet import Prophet

df = pd.read_csv('BTC-USD.csv')

df = df[["Date", "Close"]]
df.columns = ["ds","y"]
fbPropt = Prophet(yearly_seasonality=True, daily_seasonality=True)
fbPropt.fit(df)

prediction_dataframe = fbPropt.make_future_dataframe(periods=365)

prediction = fbPropt.predict(prediction_dataframe)
prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fbPropt.plot(prediction)
fig2 = fbPropt.plot_components(prediction)
