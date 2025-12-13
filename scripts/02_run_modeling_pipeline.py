import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import os

def create_features(data):
    df_feat = data.copy()
    
    # Date Features
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['is_weekend'] = (df_feat.index.weekday >= 5).astype(int)
    df_feat['day_of_month'] = df_feat.index.day
    
    # Lag Features (1 day, 2 days, 7 days)
    for lag in [1, 2, 7]:
        df_feat[f'lag_{lag}'] = df_feat['trips'].shift(lag)
    
    # Rolling Statistics (7-day window)
    df_feat['rolling_mean_7'] = df_feat['trips'].shift(1).rolling(window=7).mean()
    df_feat['rolling_std_7'] = df_feat['trips'].shift(1).rolling(window=7).std()
    
    # Holiday Flags
    holidays = pd.to_datetime(['2015-01-01', '2015-01-19', '2015-02-14', '2015-02-16'])
    df_feat['is_holiday'] = df_feat.index.isin(holidays).astype(int)
    
    # Drop NA values created by usage of lags
    df_feat.dropna(inplace=True)
    
    return df_feat

def main():
    print("Loading data...")
    df = pd.read_csv('data_clean/daily_aggregated_trips.csv', parse_dates=['date'], index_col='date')
    
    print("Feature Engineering...")
    df_features = create_features(df)
    
    test_days = 14
    train_data = df_features.iloc[:-test_days]
    test_data = df_features.iloc[-test_days:]
    
    X_train = train_data.drop(['trips', 'active_vehicles', 'weekday'], axis=1, errors='ignore')
    y_train = train_data['trips']
    X_test = test_data.drop(['trips', 'active_vehicles', 'weekday'], axis=1, errors='ignore')
    y_test = test_data['trips']
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        
        mape = mean_absolute_percentage_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results[name] = {'MAPE': mape, 'RMSE': rmse}
        print(f"{name} - MAPE: {mape:.4f}, RMSE: {rmse:.2f}")
    
    # Ensemble
    print("Building ensemble...")
    total_inv_mape = sum([1/res['MAPE'] for res in results.values()])
    weights = {name: (1/res['MAPE'])/total_inv_mape for name, res in results.items()}
    
    ensemble_pred = np.zeros_like(y_test, dtype=float)
    for name, pred in predictions.items():
        ensemble_pred += pred * weights[name]
    
    ens_mape = mean_absolute_percentage_error(y_test, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    print(f"Ensemble - MAPE: {ens_mape:.4f}, RMSE: {ens_rmse:.2f}")
    
    # Save Results
    os.makedirs('outputs', exist_ok=True)
    
    # Save Metrics
    with open('outputs/model_metrics.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        for name, res in results.items():
            f.write(f"{name} - MAPE: {res['MAPE']:.4f}, RMSE: {res['RMSE']:.2f}\n")
        f.write(f"Ensemble - MAPE: {ens_mape:.4f}, RMSE: {ens_rmse:.2f}\n")
    
    # Save Predictions
    pred_df = pd.DataFrame(predictions, index=y_test.index)
    pred_df['Ensemble'] = ensemble_pred
    pred_df['Actual'] = y_test
    pred_df.to_csv('outputs/predictions.csv')
    
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
    for name, pred in predictions.items():
        plt.plot(y_test.index, pred, label=name, linestyle='--')
    plt.plot(y_test.index, ensemble_pred, label='Ensemble', color='red', linewidth=2)
    plt.title('Trip Forecasting: Actual vs Predicted')
    plt.legend()
    plt.savefig('outputs/forecast_plot.png')
    print("Plot saved to outputs/forecast_plot.png")

if __name__ == "__main__":
    main()
