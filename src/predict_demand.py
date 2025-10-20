import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product
import warnings
import argparse
import json
import os
from metrics import calculate_metrics

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

def find_best_model(ts, p_range, d_range, q_range, s_order=(0,1,1,7), seasonal=True):
    """Find best ARIMA/SARIMA model using AIC"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    if 0 in d_range:
        d_range = [0] if adfuller(ts)[1] > 0.05 else [1]
    
    pdq_combinations = list(product(p_range, d_range, q_range))
    
    print("🔍 Searching for best model...")
    for param in pdq_combinations:
        try:
            if seasonal:
                model = SARIMAX(ts, order=param, seasonal_order=s_order)
            else:
                model = ARIMA(ts, order=param)
            
            fitted_model = model.fit(disp=False)
            
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_order = param
                best_model = fitted_model
                
        except:
            continue
    
    print(f"✅ Best model: {'SARIMA' if seasonal else 'ARIMA'} {best_order}{'x' + str(s_order) if seasonal else ''} | AIC: {best_aic:.1f}")
    return best_model, best_order

def main(input_file, horizon=90, val_days=60, outdir='outputs'):
    # Load data
    df = pd.read_csv(input_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
    ts = df['sales']
    
    # Train/validation split
    train_size = len(ts) - val_days
    train, val = ts[:train_size], ts[train_size:]
    
    print(f"📊 Data: {len(ts)} days | Train: {len(train)} | Val: {len(val)}")
    
    # Find best model
    best_model, order = find_best_model(train, [0,1,2], [0,1], [0,1,2])
    
    # Predict
    prediction = best_model.get_forecast(steps=horizon)
    prediction_mean = prediction.predicted_mean
    prediction_ci = prediction.conf_int()
    
    # Validation prediction
    val_prediction = best_model.get_forecast(steps=val_days).predicted_mean
    
    # Calculate metrics
    metrics = calculate_metrics(val.values, val_prediction.values)
    print(f"📈 Validation Metrics:")
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    # Save metrics
    with open(f'{outdir}/metrics.json', 'w') as f:
        json.dump({**metrics, 'arima_order': order, 'aic': best_model.aic}, f, indent=2)
    
    # Save prediction
    prediction_df = pd.DataFrame({
        'date': pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'),
        'prediction': prediction_mean,
        'lower_ci': prediction_ci.iloc[:, 0],
        'upper_ci': prediction_ci.iloc[:, 1]
    })
    prediction_df.to_csv(f'{outdir}/prediction.csv', index=False)
    
    # Plot 1: History vs Prediction
    plt.figure(figsize=(15, 6))
    plt.plot(ts.index, ts.values, label='Historical', color='blue', linewidth=1)
    plt.plot(val.index, val_prediction, label='Validation Prediction', color='orange', linestyle='--')
    plt.plot(prediction_df['date'], prediction_df['prediction'], label='90-day Prediction', color='red')
    plt.fill_between(prediction_df['date'], prediction_df['lower_ci'], prediction_df['upper_ci'], 
                     color='red', alpha=0.2, label='Confidence Interval')
    plt.title('Demand: History vs Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_history_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Residuals
    residuals = best_model.resid
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    residuals.plot(title='Residuals')
    plt.ylabel('Residuals')
    
    plt.subplot(2, 2, 2)
    (residuals**2).plot(title='Squared Residuals')
    plt.ylabel('Squared')
    
    plt.subplot(2, 2, 3)
    plt.stem(np.arange(len(residuals)), np.abs(residuals))
    plt.title('Absolute Residuals')
    plt.ylabel('Absolute')
    
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Outputs saved to: {outdir}/")
    print("   📊 metrics.json")
    print("   📈 prediction.csv") 
    print("   📊 fig_history_prediction.png")
    print("   🔍 fig_residuals.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/daily_sales.csv', help='Input CSV')
    parser.add_argument('--horizon', type=int, default=90, help='Prediction horizon (days)')
    parser.add_argument('--val_days', type=int, default=60, help='Validation days')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    main(args.input, args.horizon, args.val_days, args.outdir)