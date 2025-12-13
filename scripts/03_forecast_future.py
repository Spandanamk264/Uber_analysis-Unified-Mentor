import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import timedelta

def create_features(data):
    df_feat = data.copy()
    
    # Date Features
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['is_weekend'] = (df_feat.index.weekday >= 5).astype(int)
    df_feat['day_of_month'] = df_feat.index.day
    
    # Lag Features
    for lag in [1, 2, 7]:
        df_feat[f'lag_{lag}'] = df_feat['trips'].shift(lag)
    
    # Rolling Statistics
    df_feat['rolling_mean_7'] = df_feat['trips'].shift(1).rolling(window=7).mean()
    df_feat['rolling_std_7'] = df_feat['trips'].shift(1).rolling(window=7).std()
    
    # Holiday Flags (extending for March 2015 if needed, though none major there)
    holidays = pd.to_datetime(['2015-01-01', '2015-01-19', '2015-02-14', '2015-02-16'])
    df_feat['is_holiday'] = df_feat.index.isin(holidays).astype(int)
    
    return df_feat

def main():
    print("Loading data for future forecasting...")
    df = pd.read_csv('data_clean/daily_aggregated_trips.csv', parse_dates=['date'], index_col='date')
    
    # Train on Full Usage Data (up to Feb 28)
    # But first, create features to drop initial NaNs for training
    df_train_feat = create_features(df).dropna()
    
    X_train = df_train_feat.drop(['trips', 'active_vehicles'], axis=1)
    y_train = df_train_feat['trips']
    
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained on full dataset.")
    
    # Recursive Forecasting for next 7 days
    future_days = 7
    last_date = df.index.max()
    
    print(f"Forecasting {future_days} days into the future from {last_date}...")
    
    current_df = df.copy()
    # We only need 'trips' for feature calc (active_vehicles is not used as feature, only target/info)
    # Wait, active_vehicles was in the original DF but I didn't use it as feature in script 02.
    # So I can just carry 'trips' forward.
    current_df = current_df[['trips']] 
    
    future_predictions = []
    
    for i in range(1, future_days + 1):
        next_date = last_date + timedelta(days=i)
        
        # Append empty row
        current_df.loc[next_date] = np.nan
        
        # Recalculate features (this is inefficient but safe for correctness)
        df_feat_aug = create_features(current_df)
        
        # Get the feature row for the new date
        feat_row = df_feat_aug.loc[[next_date]].drop(['trips'], axis=1)
        
        # Predict
        pred = model.predict(feat_row)[0]
        
        # Fill in the prediction
        current_df.loc[next_date, 'trips'] = pred
        future_predictions.append({'date': next_date, 'predicted_trips': pred})
        
        print(f"Date: {next_date.date()}, Predicted: {pred:.2f}")
    
    # Save Future Forecasts
    forecast_df = pd.DataFrame(future_predictions)
    forecast_df.to_csv('outputs/future_forecast.csv', index=False)
    print("Future forecast saved to outputs/future_forecast.csv")

if __name__ == "__main__":
    main()
