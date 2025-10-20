import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def generate_synthetic_sales(start_date, end_date, seed=42, output_file='daily_sales.csv'):
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Base trend: linear growth
    trend = 100 + 0.5 * np.arange(n_days)
    
    # Weekly seasonality (7-day cycle)
    weekly = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Annual seasonality
    annual = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Noise
    noise = np.random.normal(0, 5, n_days)
    
    # Total sales
    sales = trend + weekly + annual + noise
    sales = np.round(np.maximum(sales, 0)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {n_days} days of synthetic sales data -> {output_file}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2023-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2024-12-31', help='End date YYYY-MM-DD')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', default='data/daily_sales.csv', help='Output file')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    generate_synthetic_sales(args.start, args.end, args.seed, args.out)