"""Royal AI — خادم بايثون: الدماغ (RandomForest + نواة الانحدار) والأخبار، ويقدّم كهف الكريستال index.html"""
import os, threading, time, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests
from flask import Flask, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# مفتاح: (رمز Binance, عقود دائمة؟, استعلام الأخبار)
ASSETS = {
    "BTC": ("BTCUSDT", False, "بتكوين OR bitcoin"), "ETH": ("ETHUSDT", False, "إيثيريوم OR ethereum"),
    "XAU": ("PAXGUSDT", False, "سعر الذهب"), "UKOIL": ("BZUSDT", True, "خام برنت"),
    "USOIL": ("CLUSDT", True, "النفط الأمريكي OR خام WTI"), "EUR": ("EURUSDT", False, "اليورو الدولار"),
}
TFS = ["15m", "1h", "4h"]
STATE = {"forecast": {}, "news": {}}

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def kernel(p, h=8.0, r=8.0, L=20):
    j = np.arange(L + 1)
    w = (1 + j ** 2 / (2 * h * h * r)) ** (-r)
    k = np.convolve(p, w)[: len(p)] / w.sum()
    k[:L] = np.nan
    return k

def klines(sym, fut, tf):
    u = "https://fapi.binance.com/fapi/v1/klines" if fut else "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(u, params={"symbol": sym, "interval": tf, "limit": 500}, timeout=10).json()
    d = pd.DataFrame(r).iloc[:, [4, 5]].astype(float)
    d.columns = ["c", "v"]
    return d

def analyze(d):
    c, v = d["c"], d["v"]
    k = pd.Series(kernel(c.to_numpy()), index=d.index)
    f = pd.DataFrame({"slope": k.diff() / c, "dist": (c - k) / c, "rsi": rsi(c),
                      "rvol": v / v.rolling(20).mean() if v.sum() > 0 else 1.0,
                      "r1": c.pct_change(), "r5": c.pct_change(5)})
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    y = (c.shift(-1) > c).astype(int).reindex(f.index)
    X, yy = f.iloc[:-1], y.iloc[:-1]
    cut = int(len(X) * 0.75)
    m = RandomForestClassifier(n_estimators=60, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=1)
    m.fit(X.iloc[:cut], yy.iloc[:cut])
    acc = float((m.predict(X.iloc[cut:]) == yy.iloc[cut:]).mean()) * 100  # اختبار صادق على بيانات لم يرَها
    m.fit(X, yy)
    pr = dict(zip(m.classes_, m.predict_proba(f.iloc[[-1]])[0]))
    return {"p": round(float(pr.get(1, 0.0)) * 100, 1), "acc": round(acc, 1), "rsi": round(float(f["rsi"].iloc[-1]), 1)}

def forecast_loop():
    while True:
        for k, (sym, fut, _) in ASSETS.items():
            for tf in TFS:
                try:
                    STATE["forecast"].setdefault(k, {})[tf] = analyze(klines(sym, fut, tf))
                except Exception as e:
                    print("forecast", k, tf, e)
        time.sleep(20)

def news_loop():
    while True:
        for k, (_, _, q) in ASSETS.items():
            try:
                r = requests.get("https://news.google.com/rss/search",
                                 params={"q": q + " when:1d", "hl": "ar", "gl": "SA", "ceid": "SA:ar"}, timeout=10)
                t = [i.findtext("title") for i in ET.fromstring(r.content).iter("item") if i.findtext("title")]
                STATE["news"][k] = [x.rsplit(" - ", 1)[0] for x in t][:8]
            except Exception as e:
                print("news", k, e)
        time.sleep(300)

@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")

@app.route("/api/forecast")
def api_forecast():
    return jsonify(STATE["forecast"])

@app.route("/api/news")
def api_news():
    return jsonify(STATE["news"])

if __name__ == "__main__":
    threading.Thread(target=forecast_loop, daemon=True).start()
    threading.Thread(target=news_loop, daemon=True).start()
    print("افتح المتصفح على: http://localhost:8000  (أو http://عنوان-الجهاز:8000 من نفس الشبكة)")
    app.run(host="0.0.0.0", port=8000, threaded=True)
