import pandas as pd
import numpy as np
import os

def process_data():
    print("Loading data...")
    df = pd.read_csv('Uber-Jan-Feb-FOIL.csv')
    
    print("Converting dates...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print("Aggregating daily trips...")
    # Aggregate by date to get daily level data
    daily_df = df.groupby('date')[['trips', 'active_vehicles']].sum().reset_index()
    daily_df.set_index('date', inplace=True)
    
    print(f"Data range: {daily_df.index.min()} to {daily_df.index.max()}")
    print(f"Total days: {len(daily_df)}")
    
    output_path = 'data_clean/daily_aggregated_trips.csv'
    # Ensure directory exists
    os.makedirs('data_clean', exist_ok=True)
    
    daily_df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    process_data()
