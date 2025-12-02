import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import time

# إعدادات الصفحة
st.set_page_config(page_title="Royal AI", page_icon="👑", layout="wide")

# ==============================================================================
# المحرك الذكي (نفس الدماغ، واجهة جديدة)
# ==============================================================================
@st.cache_resource
def get_model():
    return RandomForestClassifier(n_estimators=100, min_samples_split=10, n_jobs=1, random_state=42)

def calculate_kernel_chunk(indices, prices, h, r, lookback):
    results = {}
    prices_array = np.array(prices)
    for i in indices:
        if i < lookback: results[i] = np.nan; continue 
        current_y_sum = 0.0
        current_w_sum = 0.0
        for j in range(lookback + 1):
            if i - j < 0: break
            price = prices_array[i - j]
            w = (1 + (j**2) / (2 * h**2 * r)) ** (-r)
            current_w_sum += w
            current_y_sum += (price * w)
        if current_w_sum != 0: results[i] = current_y_sum / current_w_sum
        else: results[i] = np.nan
    return results

def get_kernel_regression_parallel(price_list, h=8.0, r=8.0, lookback=20):
    total_len = len(price_list)
    indices = range(total_len)
    chunks = np.array_split(indices, 4) # 4 أنوية كافية للويب المحلي
    results_list = Parallel(n_jobs=4, prefer="threads")(
        delayed(calculate_kernel_chunk)(chunk, price_list, h, r, lookback) for chunk in chunks
    )
    final_map = {}
    for res in results_list: final_map.update(res)
    return [final_map[i] for i in range(total_len)]

def analyze_market(ticker_symbol, tf_name, config):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=config['period'], interval=config['interval'])
        if df.empty: return None
        
        # معالجة البيانات
        if tf_name == '4h' and config['interval'] == '1h':
            ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h').apply(ohlc_dict).dropna()
        
        # المؤشرات
        price_data = df['Close'].tolist()
        df['Kernel_Line'] = get_kernel_regression_parallel(price_data)
        df['Kernel_Slope'] = df['Kernel_Line'].diff()
        df['Vol_MA'] = ta.sma(df['Volume'], length=20)
        df['RVOL'] = df['Volume'] / df['Vol_MA']
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # النقاط
        df['S_Trend'] = np.select([(df['Kernel_Slope']>0), (df['Kernel_Slope']<0)], [3, -3], 0)
        df['S_Vol'] = np.select([(df['RVOL']>1.2), (df['RVOL']<0.8)], [2, -1], 0)
        df['AI_Score'] = df['S_Trend'] + df['S_Vol']
        df.dropna(inplace=True)

        # التوقع
        features = ['Kernel_Slope', 'RSI', 'RVOL', 'AI_Score', 'Close']
        X = df[features][:-1]
        y = (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]
        
        model = get_model()
        model.fit(X, y)
        
        last_row = df[features].iloc[-1].to_frame().T
        proba = model.predict_proba(last_row)[0]
        
        return {
            'timeframe': tf_name,
            'price': last_row['Close'].values[0],
            'bull_prob': proba[1] * 100,
            'bear_prob': proba[0] * 100,
            'rvol': last_row['RVOL'].values[0]
        }
    except: return None

# ==============================================================================
# واجهة الويب
# ==============================================================================
st.title("📡 Royal System (Local Wi-Fi)")
st.markdown(f"**Last Update:** {time.strftime('%H:%M:%S')}")

TIMEFRAMES = {
    '15m': {'period': '5d', 'interval': '15m'},
    '1h':  {'period': '1mo',  'interval': '1h'},
    '4h':  {'period': '1mo',  'interval': '1h'}
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟠 Bitcoin")
    for tf, cfg in TIMEFRAMES.items():
        res = analyze_market('BTC-USD', tf, cfg)
        if res:
            color = "off"
            if res['bull_prob'] > 55: color = "normal"  # أخضر
            if res['bear_prob'] > 55: color = "inverse" # أحمر
            
            st.metric(
                label=f"{tf} | Conf: {max(res['bull_prob'], res['bear_prob']):.1f}%",
                value=f"${res['price']:.2f}",
                delta=f"RVOL: {res['rvol']:.2f}",
                delta_color=color
            )

with col2:
    st.subheader("🟡 Gold")
    for tf, cfg in TIMEFRAMES.items():
        res = analyze_market('GC=F', tf, cfg)
        if res:
            color = "off"
            if res['bull_prob'] > 55: color = "normal"
            if res['bear_prob'] > 55: color = "inverse"
            
            st.metric(
                label=f"{tf} | Conf: {max(res['bull_prob'], res['bear_prob']):.1f}%",
                value=f"${res['price']:.2f}",
                delta=f"RVOL: {res['rvol']:.2f}",
                delta_color=color
            )

# تحديث تلقائي كل 60 ثانية
time.sleep(60)
st.rerun()