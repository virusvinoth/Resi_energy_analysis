import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
def load_data(path="data/sample_energy.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    return df
def train_forecast_model(df):
    df_grouped =  df.groupby(df['timestamp'].dt.hour)["usage_kwh"].sum().reset_index()
    X = df_grouped[['timestamp']]
    y = df_grouped['usage_kwh']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def predict_usage(model, hours):
    return model.predict(np.array(hours).reshape(-1, 1))
