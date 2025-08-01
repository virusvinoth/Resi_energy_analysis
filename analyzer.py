from sklearn.ensemble import IsolationForest
def detect_anomalies(df):
    daily =df.groupby(df['timestamp'].dt.date)['usage_kwh'].sum().reset_index()
    model = IsolationForest(contamination=0.1)
    daily['anomaly'] = model.fit_predict(daily[['usage_kwh']])
    return daily

