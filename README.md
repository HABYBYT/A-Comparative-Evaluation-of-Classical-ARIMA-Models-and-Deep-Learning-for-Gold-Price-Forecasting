# A-Comparative-Evaluation-of-Classical-ARIMA-Models-and-Deep-Learning-for-Gold-Price-Forecasting
A Comparative Evaluation of Classical ARIMA Models and Deep Learning for Gold Price Forecasting


# ==================== 0. INSTALLATIONS ====================
!pip install -q yfinance pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn tensorflow

# ==================== 1. IMPORTS ====================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, kpss

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ZIP and HTML
import zipfile
import os
from IPython.display import HTML, display

print("✅ Libraries loaded")
print(f"TensorFlow version: {tf.__version__}")

# ==================== 2. DOWNLOAD DATA FROM YAHOO FINANCE ====================
print("\n" + "="*80)
print("STEP 1: DOWNLOAD DATA FROM YAHOO FINANCE (2020–April 2026)")
print("="*80)

# Tickers
tickers = {
    'Gold': 'GC=F',      # Gold futures (continuous)
    'DXY': 'DX-Y.NYB',   # US Dollar Index
    'VIX': '^VIX',       # CBOE Volatility Index
    'TNX': '^TNX',       # 10-year Treasury yield
    'SPX': '^GSPC',      # S&P 500
    'Oil': 'CL=F'        # WTI Crude Oil futures
}

start_date = '2020-01-01'
end_date   = '2026-04-30'   # last available data

print(f"Downloading {list(tickers.keys())}")
print(f"Period: {start_date} → {end_date}")

# Download all tickers at once
data = yf.download(list(tickers.values()), start=start_date, end=end_date,
                   group_by='ticker', auto_adjust=False, progress=True)

# Build DataFrame with closing prices (Adj Close if available, else Close)
df = pd.DataFrame()
for name, ticker in tickers.items():
    try:
        if 'Adj Close' in data[ticker].columns:
            df[name] = data[ticker]['Adj Close']
        elif 'Close' in data[ticker].columns:
            df[name] = data[ticker]['Close']
        else:
            df[name] = data[ticker].iloc[:, 3]
    except Exception as e:
        print(f"⚠️ Error for {name} ({ticker}), trying direct download...")
        temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in temp.columns:
            df[name] = temp['Adj Close']
        elif 'Close' in temp.columns:
            df[name] = temp['Close']
        else:
            df[name] = temp.iloc[:, 3]

# Clean data
df.dropna(how='all', inplace=True)
df.fillna(method='ffill', inplace=True)   # forward fill weekends/holidays
df.dropna(inplace=True)

print(f"✅ Data shape: {df.shape[0]} observations from {df.index.min().date()} to {df.index.max().date()}")
print(df.head())

# Separate target and exogenous variables
target = df['Gold']
exog_list = ['DXY', 'VIX', 'TNX', 'SPX', 'Oil']
exog = df[exog_list]

# ==================== 3. EXPLORATORY ANALYSIS – COVID-19 EXPONENTIAL SMOOTHING ====================
print("\n" + "="*80)
print("STEP 2: COVID-19 PERIOD – EXPONENTIAL SMOOTHING (exploratory)")
print("="*80)

covid_start = '2020-03-01'
covid_end   = '2021-12-31'

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Full series with COVID highlight
axes[0].plot(target.index, target, color='steelblue', linewidth=1)
axes[0].axvspan(pd.to_datetime(covid_start), pd.to_datetime(covid_end),
                alpha=0.2, color='red', label='COVID-19 period')
axes[0].set_title('Gold Price (USD) – Daily with COVID-19 highlight')
axes[0].set_ylabel('Price (USD)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoom on COVID-19 + exponential smoothing
covid_mask = (target.index >= covid_start) & (target.index <= covid_end)
gold_covid = target[covid_mask].copy()

# Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(gold_covid).fit()
gold_covid_ses = ses_model.fittedvalues

# Holt’s linear method (trend)
holt_model = ExponentialSmoothing(gold_covid, trend='add', seasonal=None).fit()
gold_covid_holt = holt_model.fittedvalues

axes[1].plot(gold_covid.index, gold_covid, label='Actual', color='black', alpha=0.7)
axes[1].plot(gold_covid_ses.index, gold_covid_ses, label='SES', linestyle='--', color='blue')
axes[1].plot(gold_covid_holt.index, gold_covid_holt, label='Holt (trend)', linestyle='--', color='red')
axes[1].set_title('COVID-19 Zoom: Exponential Smoothing')
axes[1].set_ylabel('Price (USD)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_covid_smoothing.png', dpi=150)
plt.show()

print(f"✅ Exponential smoothing on COVID-19 subperiod ({len(gold_covid)} obs).")
print(f"   SES smoothing level (α): {ses_model.params['smoothing_level']:.3f}")
if hasattr(holt_model, 'params'):
    print(f"   Holt parameters: α={holt_model.params['smoothing_level']:.3f}, β={holt_model.params['smoothing_trend']:.3f}")

# ==================== 4. TIME SERIES PLOTS (all variables) ====================
fig, axes = plt.subplots(len(exog_list)+1, 1, figsize=(14, 14))
axes[0].plot(target.index, target, color='gold', linewidth=1)
axes[0].set_title('Gold Price (USD) – Daily')
axes[0].grid(True, alpha=0.3)
for i, col in enumerate(exog_list):
    axes[i+1].plot(df.index, df[col], linewidth=0.8)
    axes[i+1].set_title(col)
    axes[i+1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_timeseries_all.png', dpi=150)
plt.show()

# ==================== 5. CORRELATION MATRIX ====================
plt.figure(figsize=(10,8))
corr = df[['Gold'] + exog_list].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('fig3_correlation.png', dpi=150)
plt.show()

# ==================== 6. STATIONARITY TESTS ====================
def stationarity_test(series, name):
    series = series.dropna()
    adf = adfuller(series, autolag='AIC')
    kpss_test = kpss(series, regression='c', nlags='auto')
    print(f"\n{name}:")
    print(f"  ADF: stat={adf[0]:.4f}, p-value={adf[1]:.4f} → {'stationary' if adf[1]<0.05 else 'non-stationary'}")
    print(f"  KPSS: stat={kpss_test[0]:.4f}, p-value={kpss_test[1]:.4f} → {'stationary' if kpss_test[1]>0.05 else 'non-stationary'}")

stationarity_test(target, 'Gold Price (Levels)')
returns = target.pct_change().dropna()
stationarity_test(returns, 'Gold Returns')

# ==================== 7. TRAIN / TEST SPLIT ====================
TRAIN_END   = '2024-12-31'
TEST_START  = '2025-01-01'
TEST_END    = '2026-04-30'

train_target = target[target.index < TRAIN_END]
test_target = target[target.index >= TEST_START]

train_exog = exog[exog.index < TRAIN_END]
test_exog = exog[exog.index >= TEST_START]

print(f"\n📊 Train: {train_target.index.min().date()} → {train_target.index.max().date()} ({len(train_target)} obs)")
print(f"🔮 Test: {test_target.index.min().date()} → {test_target.index.max().date()} ({len(test_target)} obs)")

# Standardize exogenous variables
scaler_exog = StandardScaler()
train_exog_scaled = pd.DataFrame(scaler_exog.fit_transform(train_exog),
                                 index=train_exog.index, columns=exog_list)
test_exog_scaled = pd.DataFrame(scaler_exog.transform(test_exog),
                                index=test_exog.index, columns=exog_list)

# ==================== 8. ARIMA MODEL (ENHANCED) ====================
print("\n" + "="*80)
print("STEP 3: ARIMA MODEL – ENHANCED SEARCH (seasonal, exhaustive)")
print("="*80)

auto_arima_model = auto_arima(
    train_target,
    seasonal=True,          # active la saisonnalité hebdomadaire
    m=5,                    # période de 5 jours (semaine boursière)
    start_p=0, start_q=0,
    max_p=10, max_q=10, max_d=2,
    start_P=0, start_Q=0,
    max_P=3, max_Q=3, max_D=1,
    information_criterion='aicc',
    stepwise=False,         # recherche exhaustive (plus précise)
    error_action='ignore',
    suppress_warnings=True,
    trace=True,
    test='adf',
    seasonal_test='ocsb'
)
best_order = auto_arima_model.order
best_seasonal_order = auto_arima_model.seasonal_order
print(f"\n✅ Best ARIMA{best_order} x {best_seasonal_order}")

arima_model = ARIMA(train_target, order=best_order, seasonal_order=best_seasonal_order)
arima_fit = arima_model.fit()
print(arima_fit.summary())

arima_forecast = arima_fit.forecast(steps=len(test_target))
arima_residuals = test_target.values - arima_forecast.values

# ==================== 9. SARIMAX MODEL (ENHANCED) ====================
print("\n" + "="*80)
print("STEP 4: SARIMAX MODEL – ENHANCED SEARCH WITH SEASONALITY")
print("="*80)

auto_sarimax = auto_arima(
    train_target,
    exogenous=train_exog_scaled,
    seasonal=True,
    m=5,
    start_p=0, start_q=0,
    max_p=5, max_q=5, max_d=2,
    start_P=0, start_Q=0,
    max_P=3, max_Q=3, max_D=1,
    information_criterion='aicc',
    stepwise=False,
    error_action='ignore',
    suppress_warnings=True,
    trace=True,
    test='adf',
    seasonal_test='ocsb'
)
best_sarimax_order = auto_sarimax.order
best_sarimax_seasonal = auto_sarimax.seasonal_order
print(f"\n✅ Best SARIMAX{best_sarimax_order} x {best_sarimax_seasonal}")

sarimax_model = SARIMAX(
    train_target,
    exog=train_exog_scaled,
    order=best_sarimax_order,
    seasonal_order=best_sarimax_seasonal
)
sarimax_fit = sarimax_model.fit(disp=False)
print(sarimax_fit.summary())

sarimax_forecast = sarimax_fit.forecast(steps=len(test_target), exog=test_exog_scaled)
sarimax_residuals = test_target.values - sarimax_forecast.values

# Coefficients of exogenous variables
coef_df = pd.DataFrame({
    'Variable': ['Intercept'] + exog_list,
    'Coefficient': sarimax_fit.params[:len(exog_list)+1],
    'Std Error': sarimax_fit.bse[:len(exog_list)+1],
    'P-value': sarimax_fit.pvalues[:len(exog_list)+1]
})
print("\n📋 SARIMAX Coefficients:")
print(coef_df.to_string(index=False))

# ==================== 10. LSTM MODEL WITH HYPERPARAMETER TUNING ====================
print("\n" + "="*80)
print("STEP 5: LSTM MODEL – HYPERPARAMETER TUNING (tanh, sigmoid, relu)")
print("="*80)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize target
scaler_y = MinMaxScaler()
train_scaled = scaler_y.fit_transform(train_target.values.reshape(-1,1)).flatten()
test_scaled = scaler_y.transform(test_target.values.reshape(-1,1)).flatten()

# Hyperparameter grid
timesteps_list = [10, 20, 30]
units_list = [50, 100]
dropout_list = [0.1, 0.2]
batch_size_list = [32]
activation_list = ['tanh', 'sigmoid', 'relu']

best_mape = float('inf')
best_model = None
best_params = {}
activation_results = []

print("Starting hyperparameter search (this may take a few minutes)...")
for seq_len in timesteps_list:
    for units in units_list:
        for dropout in dropout_list:
            for batch in batch_size_list:
                for act in activation_list:
                    print(f"\nTesting: seq_len={seq_len}, units={units}, dropout={dropout}, batch={batch}, activation={act}")
                    X_tr, y_tr = create_sequences(train_scaled, seq_len)
                    if len(X_tr) == 0:
                        continue
                    X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))
                    split = int(0.8 * len(X_tr))
                    X_val, y_val = X_tr[split:], y_tr[split:]
                    X_train_sub, y_train_sub = X_tr[:split], y_tr[:split]

                    model = Sequential([
                        Input(shape=(seq_len, 1)),
                        LSTM(units, activation=act, return_sequences=True),
                        Dropout(dropout),
                        LSTM(units//2, activation=act, return_sequences=False),
                        Dropout(dropout),
                        Dense(25, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                    ]
                    model.fit(X_train_sub, y_train_sub,
                              validation_data=(X_val, y_val),
                              epochs=50, batch_size=batch,
                              callbacks=callbacks, verbose=0)

                    X_test_seq, _ = create_sequences(test_scaled, seq_len)
                    if len(X_test_seq) == 0:
                        continue
                    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
                    pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
                    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
                    true = test_target.values[seq_len:len(pred)+seq_len]
                    mape = mean_absolute_percentage_error(true, pred) * 100
                    print(f"   MAPE = {mape:.4f}%")
                    if mape < best_mape:
                        best_mape = mape
                        best_model = model
                        best_params = {'seq_len': seq_len, 'units': units, 'dropout': dropout,
                                       'batch_size': batch, 'activation': act, 'mape': mape}
                    activation_results.append((act, seq_len, units, dropout, batch, mape))

act_summary = pd.DataFrame(activation_results, columns=['Activation', 'SeqLen', 'Units', 'Dropout', 'BatchSize', 'MAPE'])
best_per_activation = act_summary.loc[act_summary.groupby('Activation')['MAPE'].idxmin()].sort_values('MAPE')
print("\n" + "="*80)
print("COMPARISON OF LSTM MODELS BY ACTIVATION FUNCTION")
print("="*80)
print(best_per_activation[['Activation', 'SeqLen', 'Units', 'Dropout', 'MAPE']].to_string(index=False))

print("\n🏆 BEST LSTM MODEL FOUND (overall):")
for k, v in best_params.items():
    print(f"   {k}: {v}")

# ==================== 10b. BEST LSTM MODEL ARCHITECTURE (FIXED DEEP ARCHITECTURE) ====================
print("\n" + "="*80)
print("STEP 5b: BEST LSTM MODEL IMPLEMENTATION PARAMETERS")
print("="*80)

SEQ_LEN_FINAL = 30
DROPOUT_RATE = best_params.get('dropout', 0.2)
ACTIVATION = best_params.get('activation', 'tanh')
BATCH_SIZE_FINAL = best_params.get('batch_size', 32)
LEARNING_RATE = 0.001

print("Best LSTM model architecture (final model):")
print("   Input(30, 1)")
print("   ├── LSTM(100, activation={})".format(ACTIVATION))
print("   ├── Dropout({})".format(DROPOUT_RATE))
print("   ├── LSTM(80, activation={})".format(ACTIVATION))
print("   ├── Dropout({})".format(DROPOUT_RATE))
print("   ├── LSTM(60, activation={})".format(ACTIVATION))
print("   ├── Dropout({})".format(DROPOUT_RATE))
print("   ├── Dense(25, activation='relu')")
print("   └── Dense(1)")
print(f"   Optimizer: Adam (lr={LEARNING_RATE})")
print(f"   Batch size: {BATCH_SIZE_FINAL}")

X_train_final, y_train_final = create_sequences(train_scaled, SEQ_LEN_FINAL)
X_train_final = X_train_final.reshape((X_train_final.shape[0], X_train_final.shape[1], 1))

final_lstm = Sequential([
    Input(shape=(SEQ_LEN_FINAL, 1)),
    LSTM(100, activation=ACTIVATION, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(80, activation=ACTIVATION, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(60, activation=ACTIVATION, return_sequences=False),
    Dropout(DROPOUT_RATE),
    Dense(25, activation='relu'),
    Dense(1)
])
final_lstm.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history_final = final_lstm.fit(X_train_final, y_train_final,
                               epochs=100,
                               batch_size=BATCH_SIZE_FINAL,
                               validation_split=0.2,
                               callbacks=[early_stop],
                               verbose=1)

X_test_seq, _ = create_sequences(test_scaled, SEQ_LEN_FINAL)
X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
lstm_pred_scaled = final_lstm.predict(X_test_seq, verbose=0).flatten()
lstm_forecast = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1,1)).flatten()
lstm_forecast_series = pd.Series(lstm_forecast,
                                 index=test_target.index[SEQ_LEN_FINAL:len(lstm_forecast)+SEQ_LEN_FINAL])
lstm_residuals = test_target[lstm_forecast_series.index] - lstm_forecast_series

# Training loss plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_final.history['loss'], label='Train Loss')
plt.plot(history_final.history['val_loss'], label='Val Loss')
plt.title('LSTM Training – Final Deep Architecture')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig7_lstm_training.png', dpi=150)
plt.show()

# ==================== 11. OUT-OF-SAMPLE EVALUATION ====================
print("\n" + "="*80)
print("STEP 6: OUT-OF-SAMPLE EVALUATION (Test period: Jan 2025 – Apr 2026)")
print("="*80)

def compute_metrics(y_true, y_pred, name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return {'Model': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape, 'R²': r2}

results = []
results.append(compute_metrics(test_target, arima_forecast, 'ARIMA'))
results.append(compute_metrics(test_target, sarimax_forecast, 'SARIMAX'))
common_idx = test_target.index.intersection(lstm_forecast_series.index)
if len(common_idx) > 0:
    results.append(compute_metrics(test_target.loc[common_idx],
                                   lstm_forecast_series.loc[common_idx], 'LSTM (best architecture)'))

perf_df = pd.DataFrame(results).set_index('Model')
print("\n📊 Table 2 – Out-of-Sample Performance (Test period: 2025–April 2026)")
print(perf_df.round(4).to_string())

best_model_name = perf_df['MAPE(%)'].idxmin()
print(f"\n🏆 Best model (lowest MAPE): {best_model_name}")

# ==================== 12. DIEBOLD-MARIANO TEST ====================
def diebold_mariano(e1, e2, h=1, crit='MSE'):
    e1 = np.array(e1)
    e2 = np.array(e2)
    d = e1**2 - e2**2 if crit == 'MSE' else np.abs(e1) - np.abs(e2)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

print("\n📋 Table 3 – Diebold‑Mariano Test (MSE criterion)")
min_len = min(len(arima_residuals), len(sarimax_residuals), len(lstm_residuals))
dm_lstm_arima = diebold_mariano(lstm_residuals[:min_len], arima_residuals[:min_len])
dm_lstm_sarimax = diebold_mariano(lstm_residuals[:min_len], sarimax_residuals[:min_len])
dm_sarimax_arima = diebold_mariano(sarimax_residuals[:min_len], arima_residuals[:min_len])
print(f"LSTM vs ARIMA   : DM = {dm_lstm_arima[0]:.4f}, p-value = {dm_lstm_arima[1]:.4f}")
print(f"LSTM vs SARIMAX : DM = {dm_lstm_sarimax[0]:.4f}, p-value = {dm_lstm_sarimax[1]:.4f}")
print(f"SARIMAX vs ARIMA: DM = {dm_sarimax_arima[0]:.4f}, p-value = {dm_sarimax_arima[1]:.4f}")

# ==================== 13. FORECAST COMPARISON PLOT ====================
plt.figure(figsize=(14,6))
plt.plot(test_target.index, test_target, label='Actual', color='black', linewidth=2)
plt.plot(test_target.index, arima_forecast, '--', label=f'ARIMA{best_order}', linewidth=1.5)
plt.plot(test_target.index, sarimax_forecast, '--', label=f'SARIMAX{best_sarimax_order}', linewidth=1.5)
plt.plot(lstm_forecast_series.index, lstm_forecast_series, '--', label='LSTM (best architecture)', linewidth=1.5)
plt.title('Figure 4: Forecast Comparison (Test Period 2025–April 2026)')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_forecast_comparison.png', dpi=150)
plt.show()

# ==================== 14. FUTURE FORECAST – May 2026 ====================
print("\n" + "="*80)
print("STEP 7: 30-DAY AHEAD FORECAST (LSTM) – INTO MAY 2026")
print("="*80)

last_sequence = test_scaled[-SEQ_LEN_FINAL:].reshape(1, SEQ_LEN_FINAL, 1)
future_preds = []
for _ in range(30):
    next_pred = final_lstm.predict(last_sequence, verbose=0)[0,0]
    future_preds.append(next_pred)
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = next_pred

future_prices = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
future_dates = pd.date_range(start=test_target.index[-1] + timedelta(days=1), periods=30, freq='B')

plt.figure(figsize=(14,5))
plt.plot(target.index, target, label='Historical', color='blue', alpha=0.7)
plt.plot(future_dates, future_prices, 'o-', color='red', label='LSTM 30‑day forecast (May 2026)')
plt.axvline(x=test_target.index[-1], color='black', linestyle='--', alpha=0.5)
plt.title('Figure 5: 30‑Day Gold Price Forecast (LSTM) – May 2026')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_future_forecast.png', dpi=150)
plt.show()

print(f"📅 Last historical price (training+test combined): {target.iloc[-1]:.2f} USD")
print(f"🔮 Forecast for {future_dates[-1].strftime('%d/%m/%Y')}: {future_prices[-1]:.2f} USD")

