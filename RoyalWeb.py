import html, json, time, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, requests, streamlit as st, yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Royal AI", page_icon="👑", layout="wide")

ASSETS = {  # مفتاح: (رمز yfinance, الاسم, عملة)
    "BTC": ("BTC-USD", "بتكوين", "$"), "ETH": ("ETH-USD", "إيثيريوم", "$"),
    "GOLD": ("GC=F", "الذهب", "$"), "BRENT": ("BZ=F", "نفط برنت · UKOIL", "$"),
    "WTI": ("CL=F", "نفط WTI · USOIL", "$"), "EURUSD": ("EURUSD=X", "اليورو / دولار", ""),
}
TFS = {"15m": ("5d", "15m"), "1h": ("1mo", "1h"), "4h": ("1mo", "1h")}
DEF = {"assets": list(ASSETS), "cols": 2, "scale": 1.0, "bull": "#5be08a", "bear": "#ff6b6b", "glow": 0.8, "refresh": 60}
FILE = "royal_settings.json"
try:
    cfg = {**DEF, **json.load(open(FILE, encoding="utf-8"))}
except Exception:
    cfg = dict(DEF)
cfg["assets"] = [a for a in cfg["assets"] if a in ASSETS]

# ---------- لوحة التخصيص (تُحفظ تلقائياً) ----------
with st.sidebar:
    st.header("⚙️ التخصيص")
    cfg["assets"] = st.multiselect("الأدوات المعروضة", list(ASSETS), cfg["assets"], format_func=lambda k: ASSETS[k][1])
    cfg["cols"] = st.slider("عدد الأعمدة", 1, 3, int(cfg["cols"]))
    cfg["scale"] = st.slider("حجم البطاقات", 0.7, 1.6, float(cfg["scale"]), 0.05)
    a, b = st.columns(2)
    cfg["bull"] = a.color_picker("لون الصعود", cfg["bull"])
    cfg["bear"] = b.color_picker("لون الهبوط", cfg["bear"])
    cfg["glow"] = st.slider("قوة التوهج", 0.0, 1.0, float(cfg["glow"]), 0.05)
    cfg["refresh"] = st.slider("التحديث (ثانية)", 30, 300, int(cfg["refresh"]), 10)
    try:
        json.dump(cfg, open(FILE, "w", encoding="utf-8"))
    except Exception:
        pass

# ---------- التأثيرات البصرية ----------
st.markdown(f"<style>.stApp{{--g:{cfg['glow']};--s:{cfg['scale']}}}</style>", unsafe_allow_html=True)
st.markdown("""<style>
html,body{background:#000!important}
[data-testid=stAppViewContainer],[data-testid=stMain],.stMain,.main{background:transparent!important}
header[data-testid=stHeader]{background:transparent!important}
.stApp{direction:rtl;color:#fff;
 background:radial-gradient(55% 38% at 15% 0,rgba(30,143,80,var(--g)),transparent 70%) 0 0/200% 100% no-repeat,
 radial-gradient(55% 38% at 50% 0,rgba(66,133,244,var(--g)),transparent 70%) 0 0/200% 100% no-repeat,
 radial-gradient(55% 38% at 85% 0,rgba(155,114,203,var(--g)),transparent 70%) 0 0/200% 100% no-repeat,#000;
 animation:flow 14s ease-in-out infinite alternate}
@keyframes flow{from{background-position:0 0,0 0,0 0}to{background-position:100% 0,100% 0,100% 0}}
.edge{position:fixed;inset:0;pointer-events:none;z-index:1;opacity:calc(.65*var(--g));
 background:linear-gradient(90deg,#af87eb,#8c8c9e,#f0f0fa,#9169d7,#78788a,#af87eb) 0 0/100vw 100% repeat-x;
 -webkit-mask-image:linear-gradient(#000,transparent 9%,transparent 91%,#000);mask-image:linear-gradient(#000,transparent 9%,transparent 91%,#000);
 animation:slide 12s linear infinite}
@keyframes slide{to{background-position:100vw 0}}
h1.t,.card,.psy,.v,.tick,.ld,.cap{position:relative;z-index:2;color:#fff!important}
h1.t{font-weight:200;margin:0 0 4px;text-shadow:0 0 26px #b9a2ff}
.ld{padding:12px 16px;margin:8px 0;border-radius:14px;background:rgba(255,255,255,.07);border:1px solid rgba(185,162,255,.35)}
.cap{opacity:.55;font-size:.8em;margin-top:8px}
.card{margin:0 0 14px;padding:calc(14px*var(--s));border-radius:20px;font-size:calc(15px*var(--s));background:rgba(255,255,255,.06);
 backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.12);box-shadow:0 0 calc(34px*var(--g)) -8px var(--c);animation:pulse 5s ease-in-out infinite}
@keyframes pulse{50%{box-shadow:0 0 calc(54px*var(--g)) -4px var(--c)}}
.h{display:flex;justify-content:space-between;align-items:center;gap:10px}.h svg{max-width:44%}
.p{direction:ltr;text-align:right;font-size:1.9em;font-weight:250;color:var(--c)!important}
.row{display:grid;grid-template-columns:2.4em 1fr 3.8em;align-items:center;gap:8px;margin-top:6px;font-size:.85em}
.row small{grid-column:1/-1;opacity:.55;font-size:.72em}
.bar{height:6px;border-radius:6px;background:rgba(255,255,255,.12);overflow:hidden;direction:ltr}.bar i{display:block;height:100%}
.psy{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:8px 0}
.psy .b{padding:12px;border-radius:16px;background:rgba(255,255,255,.06);border:1px solid rgba(185,162,255,.3);box-shadow:0 0 28px -10px #b9a2ff;text-align:center}
.psy small,.psy span{display:block;opacity:.65;font-size:.75em}.psy b{font-size:1.5em;font-weight:250}
.v{margin:0 0 12px;opacity:.9;font-size:.9em}
.tick{overflow:hidden;white-space:nowrap;margin:6px 0 14px;padding:8px 0;border-radius:12px;background:rgba(255,255,255,.05);direction:ltr}
.tick span{display:inline-block;padding-left:100%;animation:mq 90s linear infinite;font-weight:300;font-size:.9em}
@keyframes mq{to{transform:translateX(-100%)}}
</style>""", unsafe_allow_html=True)

# ---------- المحرك (مصحّح) ----------
def rsi(s, n=14):  # RSI بطريقة Wilder بدون pandas_ta
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def kernel(p, h=8.0, r=8.0, L=20):  # نفس النواة، لكن بمتجهات بدل الحلقات المتداخلة
    j = np.arange(L + 1)
    w = (1 + j ** 2 / (2 * h * h * r)) ** (-r)
    k = np.convolve(p, w)[: len(p)] / w.sum()
    k[:L] = np.nan
    return k

@st.cache_data(ttl=120, show_spinner=False)
def hist(sym, period, interval):
    return yf.Ticker(sym).history(period=period, interval=interval, timeout=15)

@st.cache_data(ttl=120, show_spinner=False)
def analyze(sym, tf):
    try:
        df = hist(sym, *TFS[tf])
        if df.empty:
            return {"err": "لا توجد بيانات"}
        if tf == "4h":
            df = df.resample("4h").agg({"Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
        c, v = df["Close"], df["Volume"]
        has_vol = v.sum() > 0  # الفوركس بلا حجم تداول
        k = pd.Series(kernel(c.to_numpy(float)), index=df.index)
        f = pd.DataFrame({"slope": k.diff() / c, "dist": (c - k) / c, "rsi": rsi(c),
                          "rvol": v / v.rolling(20).mean() if has_vol else 1.0,
                          "r1": c.pct_change(), "r5": c.pct_change(5)})
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        if len(f) < 80:
            return {"err": "بيانات غير كافية"}
        y = (c.shift(-1) > c).astype(int).reindex(f.index)
        X, yy = f.iloc[:-1], y.iloc[:-1]
        cut = int(len(X) * 0.75)
        m = RandomForestClassifier(n_estimators=50, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=1)
        m.fit(X.iloc[:cut], yy.iloc[:cut])
        acc = float((m.predict(X.iloc[cut:]) == yy.iloc[cut:]).mean()) * 100  # اختبار صادق على بيانات لم يرَها
        m.fit(X, yy)
        pr = dict(zip(m.classes_, m.predict_proba(f.iloc[[-1]])[0]))
        return {"price": float(c.iloc[-1]), "bull": float(pr.get(1, 0.0)) * 100, "acc": acc,
                "rsi": float(f["rsi"].iloc[-1]), "spark": [float(x) for x in c.iloc[-40:]]}
    except Exception as e:
        return {"err": str(e)[:70]}

@st.cache_data(ttl=600, show_spinner=False)
def fear_greed():
    try:
        j = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6).json()["data"][0]
        return int(j["value"]), j["value_classification"]
    except Exception:
        return None, None

@st.cache_data(ttl=300, show_spinner=False)
def news():
    try:
        q = "النفط OR الذهب OR بتكوين OR إيثيريوم OR اليورو OR عاجل when:1d"
        r = requests.get("https://news.google.com/rss/search", params={"q": q, "hl": "ar", "gl": "SA", "ceid": "SA:ar"}, timeout=8)
        return [i.findtext("title") for i in ET.fromstring(r.content).iter("item") if i.findtext("title")][:12]
    except Exception:
        return []

# ---------- العرض ----------
def spark(vals, col):
    lo, hi = min(vals), max(vals)
    sp = (hi - lo) or 1
    pts = " ".join(f"{i * 140 / (len(vals) - 1):.1f},{30 - (v - lo) / sp * 28:.1f}" for i, v in enumerate(vals))
    return f'<svg viewBox="0 0 140 30" width="100%" height="30"><polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/></svg>'

def card(key):
    sym, name, cur = ASSETS[key]
    rs = {tf: analyze(sym, tf) for tf in TFS}
    ok = {k: r for k, r in rs.items() if "err" not in r}
    if not ok:
        err = next(iter(rs.values())).get("err", "")
        return f'<div class="card" style="--c:#888"><b>{name}</b><div>⚠ {html.escape(err)}</div></div>', [], None
    base = ok.get("1h") or next(iter(ok.values()))
    sigs = [float(np.clip((r["bull"] - 50) / 10, -1, 1)) for r in ok.values()]
    tot = float(np.mean(sigs))
    col = cfg["bull"] if tot > 0.15 else cfg["bear"] if tot < -0.15 else "#b9b9cc"
    rows = ""
    for tf, r in ok.items():
        b = r["bull"]
        c2 = cfg["bull"] if b >= 50 else cfg["bear"]
        warn = "" if r["acc"] >= 55 else " ⚠ قريب من العشوائية"
        rows += (f'<div class="row"><span>{tf}</span><div class="bar"><i style="width:{b:.0f}%;background:{c2}"></i></div>'
                 f'<span>{max(b, 100 - b):.0f}% {"▲" if b >= 50 else "▼"}</span><small>اختبار صادق {r["acc"]:.0f}%{warn}</small></div>')
    p = base["price"]
    ps = f"{p:,.4f}" if p < 10 else f"{p:,.2f}"
    h = f'<div class="card" style="--c:{col}"><div class="h"><b>{name}</b>{spark(base["spark"], col)}</div><div class="p">{cur}{ps}</div>{rows}</div>'
    return h, sigs, abs(base["rsi"] - 50) / 50

def psy(sigs, heats):
    if not sigs:
        return ""
    fg, fgt = fear_greed()
    ar = {"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", "Greed": "طمع", "Extreme Greed": "طمع شديد"}
    maj = 1 if sum(sigs) >= 0 else -1
    agree = float(np.mean([np.sign(s) == maj for s in sigs if s != 0] or [0])) * 100
    heat = float(np.mean(heats)) * 100 if heats else 0
    box = lambda t, v, s: f'<div class="b"><small>{t}</small><b>{v}</b><span>{s}</span></div>'
    if agree >= 70 and heat >= 45:
        v = "إجماع حاد مع تطرّف: الجمهور مندفع في اتجاه واحد، وهذه الحالات كثيراً ما تسبق انعكاساً."
    elif agree >= 70:
        v = "فكر جماعي موحّد: معظم الأطر والأدوات تشير لنفس الاتجاه دون تطرّف كبير."
    else:
        v = "آراء متفرقة: لا يوجد قطيع واضح الآن."
    if fg is not None and (fg <= 25 or fg >= 75):
        v += " ومؤشر الخوف والطمع في منطقة متطرفة."
    fgv = f"{fg} · {ar.get(fgt, fgt)}" if fg is not None else "—"
    return ('<div class="psy">' + box("🧠 خوف/طمع (كريبتو)", fgv, "مزاج الجمهور") +
            box("🤝 توافق الإشارات", f'{agree:.0f}% {"صعود" if maj > 0 else "هبوط"}', "فكر مشترك بين الأدوات") +
            box("🔥 حرارة القطيع", f"{heat:.0f}%", "ابتعاد RSI عن الحياد") + f'</div><div class="v">{v} (مقياس للمزاج لا دليل سببي)</div>')

st.markdown('<div class="edge"></div><h1 class="t">👑 Royal AI</h1>', unsafe_allow_html=True)

def dash():
    top = st.empty()
    tk = st.empty()
    cols = st.columns(cfg["cols"])
    slots = [cols[i % cfg["cols"]].empty() for i in range(len(cfg["assets"]))]
    top.markdown('<div class="ld">⏳ جارٍ تحليل الأسواق… قد يستغرق أول تحميل دقيقة أو أكثر</div>', unsafe_allow_html=True)
    nw = news()
    if nw:
        tk.markdown('<div class="tick"><span>' + " &nbsp;•&nbsp; ".join(html.escape(t) for t in nw) + "</span></div>", unsafe_allow_html=True)
    sigs, heats = [], []
    for n, (k, slot) in enumerate(zip(cfg["assets"], slots), 1):
        h, s_, ht = card(k)
        slot.markdown(h, unsafe_allow_html=True)
        sigs += s_
        if ht is not None:
            heats.append(ht)
        top.markdown(f'<div class="ld">⏳ تم تحليل {n} من {len(slots)}</div>', unsafe_allow_html=True)
    top.markdown(psy(sigs, heats), unsafe_allow_html=True)
    st.markdown(f'<div class="cap">آخر تحديث {time.strftime("%H:%M:%S")} · «اختبار صادق» = صحة النموذج على بيانات لم يتدرّب عليها، وقرب 50% يعني قرب العشوائية. ليست نصيحة مالية.</div>', unsafe_allow_html=True)

(st.fragment(run_every=cfg["refresh"])(dash) if hasattr(st, "fragment") else dash)()
