def generate_recommendations(df):
    recs = []
    if df['usage_kwh'].mean() > 1.5:
        recs.append("Consider using high-power devices during off-peak hours.")
        if df['usage_kwh'].max() > 3:
            recs.append("Large spikes detected – check air conditioning usage.")
    return recs