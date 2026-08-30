# -*- coding: utf-8 -*-
"""
داشبورد مانیتورینگ حرفه‌ای مدل طبقه‌بندی متون جرم/حقوقی (Crime & Legal Text Classification)
ساخته شده با Streamlit + TensorFlow/Keras + Plotly
"""

import os
import pickle
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import tensorflow as tf

# ────────────────────────────────────────────────────────────────────────────
# تنظیمات پایه صفحه
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="داشبورد مانیتورینگ مدل | Crime Text AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "vectorizer_vocab_crime_model.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "BiLSTM": "model_BiLSTM_crime.keras",
    "LSTM": "model_LSTM_crime.keras",
    "Conv1D": "model_Conv1D_crime.keras",
    "Dense": "model_Dense_crime.keras",
}

SEQ_LEN = 32
MAX_TOKENS = 5000

LABEL_NAMES = {
    0: "بدون جرم",
    1: "جرم خشونت‌آمیز / مواد مخدر",
    2: "سرقت / جرم سایبری",
    3: "فساد اداری / مالی",
}
LABEL_COLORS = {
    0: "#22c55e",
    1: "#ef4444",
    2: "#f59e0b",
    3: "#8b5cf6",
}
CLASS_IDS = [0, 1, 2, 3]

# ────────────────────────────────────────────────────────────────────────────
# استایل مدرن و مینیمال (تم تیره + فونت وزیرمتن + راست‌چین)
# ────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Vazirmatn', sans-serif !important;
}
.block-container {
    direction: rtl;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
p, li, span, label, .stMarkdown, div[data-testid="stMetricLabel"] {
    text-align: right;
}
h1, h2, h3, h4 {
    text-align: right;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] {
    direction: rtl;
    border-left: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { text-align: right; }

/* هدر گرادیانی */
.hero {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 45%, #ec4899 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(99,102,241,0.25);
}
.hero h1 {
    color: white !important;
    font-size: 1.65rem;
    margin: 0 0 6px 0;
}
.hero p {
    color: rgba(255,255,255,0.9);
    margin: 0;
    font-size: 0.95rem;
}

/* کارت‌های شیشه‌ای مدرن */
.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(6px);
    margin-bottom: 14px;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 4px;
    display:flex; align-items:center; gap:8px;
}
.section-sub {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    margin-bottom: 14px;
}

/* بج هشدار */
.badge {
    display:inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
}
.badge-red { background: rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.35);}
.badge-yellow { background: rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.35);}
.badge-green { background: rgba(34,197,94,0.15); color:#4ade80; border:1px solid rgba(34,197,94,0.35);}
.badge-gray { background: rgba(148,163,184,0.15); color:#cbd5e1; border:1px solid rgba(148,163,184,0.35);}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 12px 16px 6px 16px;
}
div[data-testid="stMetricValue"] { font-size: 1.5rem; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
}
hr { border-color: rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#8b5cf6"


# ────────────────────────────────────────────────────────────────────────────
# بارگذاری داده، وکتورایزر و مدل‌ها (کش می‌شوند)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="در حال بارگذاری وکتورایزر...")
def load_vectorizer():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab = [str(v) for v in vocab]
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    # دو توکن اول (خالی و [UNK]) رزرو هستند و به‌صورت خودکار اضافه می‌شوند
    vectorizer.set_vocabulary(vocab[2:])
    return vectorizer, vocab


@st.cache_resource(show_spinner="در حال بارگذاری مدل‌ها...")
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        models[name] = tf.keras.models.load_model(path, compile=False)
    return models


@st.cache_data(show_spinner="در حال آماده‌سازی داده‌ها...")
def load_and_split_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    return (
        df.reset_index(drop=True),
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def vectorize_texts(vectorizer, texts):
    return vectorizer(tf.constant(texts)).numpy()


@st.cache_data(show_spinner=False)
def predict_batch(model_name, texts_tuple):
    """پیش‌بینی احتمالات کلاس برای یک دسته متن با یک مدل مشخص"""
    texts = list(texts_tuple)
    vectorizer, _ = load_vectorizer()
    models = load_models()
    X = vectorize_texts(vectorizer, texts)
    probs = models[model_name].predict(X, verbose=0)
    return probs


def enrich_with_predictions(df, model_name):
    probs = predict_batch(model_name, tuple(df["text"].tolist()))
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    out = df.copy().reset_index(drop=True)
    out["pred_label"] = pred
    out["confidence"] = conf
    out["correct"] = out["pred_label"] == out["label"]
    for c in CLASS_IDS:
        out[f"prob_{c}"] = probs[:, c]
    out["token_len"] = out["text"].str.split().apply(len)
    return out


@st.cache_data(show_spinner=False)
def simulate_production_log(model_name, seed=7, n_events=1400, days_back=14):
    """
    شبیه‌سازی یک لاگ تولید (production traffic) با ری‌سمپل کردن از کل دیتاست
    و اختصاص زمان‌های تصادفی طی N روز گذشته — برای نمایش توان عملیاتی،
    تاریخچه هشدار و روند انحراف داده در طول زمان.
    """
    full_df, _, _, _ = load_and_split_data()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(full_df), size=n_events)
    sample = full_df.iloc[idx].reset_index(drop=True)

    enriched = enrich_with_predictions(sample, model_name)

    now = datetime(2026, 8, 30, 18, 0, 0)
    # وزن‌دهی به ساعات کاری برای واقعی‌تر شدن نمودار توان عملیاتی
    hour_weights = np.array([
        0.2, 0.15, 0.1, 0.1, 0.15, 0.3, 0.6, 1.0, 1.4, 1.7, 1.9, 1.8,
        1.6, 1.7, 1.9, 1.8, 1.6, 1.3, 1.1, 0.9, 0.7, 0.5, 0.35, 0.25
    ])
    hour_weights = hour_weights / hour_weights.sum()

    minutes_back = rng.integers(0, days_back * 24 * 60, size=len(enriched))
    day_offset = minutes_back // (24 * 60)
    chosen_hour = rng.choice(24, size=len(enriched), p=hour_weights)
    chosen_min = rng.integers(0, 60, size=len(enriched))

    timestamps = []
    for d, h, m in zip(day_offset, chosen_hour, chosen_min):
        ts = now - timedelta(days=int(d))
        ts = ts.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        timestamps.append(ts)

    # کمی انحراف مصنوعی داده در ۲ روز اخیر: متن‌های کمی طولانی‌تر/کوتاه‌تر
    enriched["timestamp"] = timestamps
    enriched = enriched.sort_values("timestamp").reset_index(drop=True)
    return enriched


def request_id(text, ts):
    return hashlib.md5((text + str(ts)).encode()).hexdigest()[:8]


# ────────────────────────────────────────────────────────────────────────────
# بارگذاری اولیه
# ────────────────────────────────────────────────────────────────────────────
try:
    vectorizer, vocab_list = load_vectorizer()
    models = load_models()
    full_df, train_df, val_df, test_df = load_and_split_data()
    LOAD_OK = True
except Exception as e:
    LOAD_OK = False
    st.error(f"خطا در بارگذاری مدل‌ها یا داده‌ها: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# سایدبار
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Crime Text AI")
    st.caption("سامانه مانیتورینگ مدل طبقه‌بندی متون جرم/حقوقی")
    st.markdown("---")

    model_name = st.selectbox(
        "🧠 انتخاب مدل فعال",
        list(MODEL_FILES.keys()),
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📊 Performance Metrics")
    st.caption(f"ارزیابی روی مجموعهٔ تست (نگهداری‌شده) — {len(test_df)} نمونه")

    test_eval = enrich_with_predictions(test_df, model_name)
    acc = accuracy_score(test_eval["label"], test_eval["pred_label"])
    prec = precision_score(test_eval["label"], test_eval["pred_label"], average="weighted", zero_division=0)
    rec = recall_score(test_eval["label"], test_eval["pred_label"], average="weighted", zero_division=0)
    f1 = f1_score(test_eval["label"], test_eval["pred_label"], average="weighted", zero_division=0)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc*100:.1f}%")
    c2.metric("F1-Score", f"{f1*100:.1f}%")
    c3, c4 = st.columns(2)
    c3.metric("Precision", f"{prec*100:.1f}%")
    c4.metric("Recall", f"{rec*100:.1f}%")

    st.markdown("---")
    st.caption("🕒 آخرین بروزرسانی: اکنون")
    st.caption("🟢 وضعیت سرویس: آنلاین")

# لاگ شبیه‌سازی‌شده تولید (برای توان عملیاتی / هشدارها / روند زمانی)
prod_log = simulate_production_log(model_name)

# ────────────────────────────────────────────────────────────────────────────
# هدر
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <h1>داشبورد مانیتورینگ مدل طبقه‌بندی متون جرم و حقوقی</h1>
        <p>مدل فعال: <b>{model_name}</b> · نظارت لحظه‌ای بر عملکرد، کیفیت داده و رفتار مدل در تولید</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📈 نمای کلی",
    "🧩 ماتریس درهم‌ریختگی",
    "🌊 انحراف داده",
    "🔍 تحلیل خطا",
    "🚨 تاریخچه هشدار",
    "🗂️ فیلتر و خروجی داده",
    "💡 تفسیرپذیری",
])

# ════════════════════════════════════════════════════════════════════════════
# تب ۱ — نمای کلی (Performance + Throughput)
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📈 خلاصه عملکرد مدل</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">معیارهای کلیدی روی مجموعهٔ تست + وضعیت ترافیک تولید</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{acc*100:.1f}%")
    k2.metric("Precision", f"{prec*100:.1f}%")
    k3.metric("Recall", f"{rec*100:.1f}%")
    k4.metric("F1-Score", f"{f1*100:.1f}%")
    avg_conf = prod_log["confidence"].mean()
    k5.metric("میانگین اطمینان (تولید)", f"{avg_conf*100:.1f}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    colA, colB = st.columns([1.4, 1])
    with colA:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Throughput — تعداد پیش‌بینی در ساعت</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">ترافیک شبیه‌سازی‌شدهٔ تولید طی ۱۴ روز اخیر (بر اساس ساعت روز)</div>', unsafe_allow_html=True)

        hourly = prod_log.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly_count = hourly.groupby("hour").size().reindex(range(24), fill_value=0)
        fig = px.bar(
            x=hourly_count.index, y=hourly_count.values,
            labels={"x": "ساعت روز", "y": "تعداد پیش‌بینی"},
            template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        total_events = len(prod_log)
        span_hours = (prod_log["timestamp"].max() - prod_log["timestamp"].min()).total_seconds() / 3600
        avg_per_hour = total_events / max(span_hours, 1)
        st.caption(f"📦 مجموع پیش‌بینی‌های ثبت‌شده: **{total_events:,}** · میانگین توان عملیاتی: **{avg_per_hour:.1f} پیش‌بینی/ساعت**")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏷️ توزیع کلاس‌های پیش‌بینی‌شده</div>', unsafe_allow_html=True)
        dist = prod_log["pred_label"].value_counts().reindex(CLASS_IDS, fill_value=0)
        fig2 = px.pie(
            names=[LABEL_NAMES[c] for c in dist.index], values=dist.values,
            color=[LABEL_NAMES[c] for c in dist.index],
            color_discrete_map={LABEL_NAMES[c]: LABEL_COLORS[c] for c in CLASS_IDS},
            template=PLOTLY_TEMPLATE, hole=0.55,
        )
        fig2.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                            legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕓 آخرین پیش‌بینی‌های ثبت‌شده</div>', unsafe_allow_html=True)
    recent = prod_log.sort_values("timestamp", ascending=False).head(8)[
        ["timestamp", "text", "pred_label", "confidence"]
    ].copy()
    recent["کلاس پیش‌بینی‌شده"] = recent["pred_label"].map(LABEL_NAMES)
    recent["اطمینان"] = (recent["confidence"] * 100).round(1).astype(str) + "%"
    recent = recent.rename(columns={"timestamp": "زمان", "text": "متن"})[
        ["زمان", "متن", "کلاس پیش‌بینی‌شده", "اطمینان"]
    ]
    st.dataframe(recent, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# تب ۲ — ماتریس درهم‌ریختگی
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🧩 ماتریس درهم‌ریختگی (Confusion Matrix)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">عملکرد مدل روی مجموعهٔ تست به تفکیک کلاس واقعی و پیش‌بینی‌شده</div>', unsafe_allow_html=True)

    cm = confusion_matrix(test_eval["label"], test_eval["pred_label"], labels=CLASS_IDS)
    labels_fa = [LABEL_NAMES[c] for c in CLASS_IDS]

    normalize = st.toggle("نرمال‌سازی بر اساس درصد", value=False)
    if normalize:
        cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        text_fmt = [[f"{v*100:.1f}%" for v in row] for row in cm_show]
        zmax = 1
    else:
        cm_show = cm
        text_fmt = [[str(v) for v in row] for row in cm]
        zmax = None

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_show, x=labels_fa, y=labels_fa,
        text=text_fmt, texttemplate="%{text}",
        colorscale="Purples", zmax=zmax,
    ))
    fig_cm.update_layout(
        template=PLOTLY_TEMPLATE, height=480,
        xaxis_title="کلاس پیش‌بینی‌شده", yaxis_title="کلاس واقعی",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_cm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("📋 گزارش تفصیلی به تفکیک کلاس"):
        rows = []
        for c in CLASS_IDS:
            mask = test_eval["label"] == c
            p = precision_score(test_eval["label"], test_eval["pred_label"], labels=[c], average="micro", zero_division=0) if mask.sum() else 0
            rows.append({
                "کلاس": LABEL_NAMES[c],
                "تعداد نمونه": int(mask.sum()),
                "دقت پیش‌بینی صحیح": f"{(test_eval[mask]['correct'].mean()*100 if mask.sum() else 0):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# تب ۳ — انحراف داده (Data Drift Detection + Over Time)
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🌊 تشخیص انحراف داده (Data Drift Detection)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">مقایسهٔ توزیع داده‌های ورودی جدید (تولید) با توزیع مرجع (داده آموزش)</div>', unsafe_allow_html=True)

    ref_len = train_df["text"].str.split().apply(len)
    prod_len = prod_log["token_len"]

    ks_stat, ks_p = ks_2samp(ref_len, prod_len)
    drift_flag = ks_p < 0.05

    d1, d2, d3 = st.columns(3)
    d1.metric("میانگین طول متن (مرجع)", f"{ref_len.mean():.1f} توکن")
    d2.metric("میانگین طول متن (تولید)", f"{prod_len.mean():.1f} توکن")
    d3.metric("آماره KS", f"{ks_stat:.3f}", delta=("⚠️ انحراف معنادار" if drift_flag else "✅ پایدار"))

    colD1, colD2 = st.columns(2)
    with colD1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">توزیع طول متن: مرجع در برابر تولید</div>', unsafe_allow_html=True)
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=ref_len, name="داده مرجع (Train)", opacity=0.6,
                                      marker_color="#6366f1", histnorm="probability"))
        fig_d.add_trace(go.Histogram(x=prod_len, name="داده تولید (Production)", opacity=0.6,
                                      marker_color="#ec4899", histnorm="probability"))
        fig_d.update_layout(barmode="overlay", template=PLOTLY_TEMPLATE, height=360,
                             margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colD2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">توزیع کلاس‌ها: مرجع در برابر تولید</div>', unsafe_allow_html=True)
        ref_dist = train_df["label"].value_counts(normalize=True).reindex(CLASS_IDS, fill_value=0)
        prod_dist = prod_log["pred_label"].value_counts(normalize=True).reindex(CLASS_IDS, fill_value=0)
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=labels_fa, y=ref_dist.values, name="مرجع", marker_color="#6366f1"))
        fig_c.add_trace(go.Bar(x=labels_fa, y=prod_dist.values, name="تولید", marker_color="#ec4899"))
        fig_c.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=360,
                             margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 روند انحراف داده در طول زمان (Data Drift Over Time)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">میانگین روزانهٔ طول توکن متن‌های ورودی در برابر خط پایهٔ داده مرجع</div>', unsafe_allow_html=True)

    daily = prod_log.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_len = daily.groupby("date")["token_len"].mean().reset_index()

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=daily_len["date"], y=daily_len["token_len"], mode="lines+markers",
        name="میانگین طول توکن (روزانه)", line=dict(color="#f59e0b", width=3),
        marker=dict(size=7),
    ))
    fig_t.add_hline(y=ref_len.mean(), line_dash="dash", line_color="#6366f1",
                     annotation_text="خط پایه (مرجع)", annotation_position="top left")
    fig_t.update_layout(template=PLOTLY_TEMPLATE, height=340,
                         xaxis_title="تاریخ", yaxis_title="میانگین طول توکن",
                         margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# تب ۴ — تحلیل خطا
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🔍 تحلیل خطا (Error Analysis)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">نمونه‌هایی که مدل با اطمینان پایین یا به‌اشتباه پیش‌بینی کرده است</div>', unsafe_allow_html=True)

    thresh = st.slider("آستانهٔ اطمینان برای برچسب‌گذاری «اطمینان پایین»", 0.0, 1.0, 0.6, 0.05)

    low_conf = test_eval[test_eval["confidence"] < thresh].copy()
    wrong = test_eval[~test_eval["correct"]].copy()

    e1, e2, e3 = st.columns(3)
    e1.metric("نمونه‌های اطمینان پایین", len(low_conf))
    e2.metric("پیش‌بینی‌های نادرست", len(wrong))
    e3.metric("نرخ خطای کلی", f"{(1-acc)*100:.1f}%")

    view = st.radio("نمایش:", ["اطمینان پایین", "پیش‌بینی نادرست", "هر دو"], horizontal=True)
    if view == "اطمینان پایین":
        show_df = low_conf
    elif view == "پیش‌بینی نادرست":
        show_df = wrong
    else:
        show_df = test_eval[(test_eval["confidence"] < thresh) | (~test_eval["correct"])]

    show_df = show_df.sort_values("confidence").copy()
    show_df["برچسب واقعی"] = show_df["label"].map(LABEL_NAMES)
    show_df["برچسب پیش‌بینی"] = show_df["pred_label"].map(LABEL_NAMES)
    show_df["اطمینان"] = (show_df["confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(
        show_df.rename(columns={"text": "متن"})[["متن", "برچسب واقعی", "برچسب پیش‌بینی", "اطمینان"]],
        use_container_width=True, hide_index=True, height=420,
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">اطمینان مدل به تفکیک صحت پیش‌بینی</div>', unsafe_allow_html=True)
    fig_e = px.box(
        test_eval, x="correct", y="confidence", color="correct",
        color_discrete_map={True: "#22c55e", False: "#ef4444"},
        labels={"correct": "پیش‌بینی صحیح بوده؟", "confidence": "اطمینان مدل"},
        template=PLOTLY_TEMPLATE,
    )
    fig_e.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_e, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# تب ۵ — تاریخچه هشدار
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🚨 تاریخچه هشدارها (Alert History)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">هشدارهای خودکار تولیدشده بر اساس افت دقت، اطمینان پایین یا انحراف داده در بازه‌های روزانه</div>', unsafe_allow_html=True)

    daily_log = prod_log.copy()
    daily_log["date"] = daily_log["timestamp"].dt.date
    alerts = []
    for date, g in daily_log.groupby("date"):
        day_acc = g["correct"].mean()
        day_conf = g["confidence"].mean()
        day_len = g["token_len"].mean()
        drift_stat, drift_p = ks_2samp(ref_len, g["token_len"]) if len(g) > 3 else (0, 1)

        if day_acc < 0.55:
            alerts.append((date, "افت دقت مدل", f"دقت روزانه به {day_acc*100:.1f}% رسید", "قرمز"))
        elif day_acc < 0.68:
            alerts.append((date, "افت دقت مدل", f"دقت روزانه به {day_acc*100:.1f}% رسید", "زرد"))

        if day_conf < 0.55:
            alerts.append((date, "اطمینان پایین", f"میانگین اطمینان مدل {day_conf*100:.1f}% بود", "قرمز"))
        elif day_conf < 0.65:
            alerts.append((date, "اطمینان پایین", f"میانگین اطمینان مدل {day_conf*100:.1f}% بود", "زرد"))

        if drift_p < 0.01:
            alerts.append((date, "انحراف داده", f"تغییر معنادار در طول متن‌های ورودی (KS={drift_stat:.2f})", "قرمز"))
        elif drift_p < 0.05:
            alerts.append((date, "انحراف داده", f"تغییر جزئی در توزیع داده ورودی (KS={drift_stat:.2f})", "زرد"))

        if len(g) > daily_log.groupby("date").size().mean() * 1.6:
            alerts.append((date, "افزایش ترافیک", f"{len(g)} درخواست در این روز ثبت شد (بیش از حد معمول)", "زرد"))

    alerts_df = pd.DataFrame(alerts, columns=["تاریخ", "نوع هشدار", "پیام", "شدت"])
    alerts_df = alerts_df.sort_values("تاریخ", ascending=False)

    if alerts_df.empty:
        st.success("✅ در بازهٔ اخیر هیچ هشداری ثبت نشده است — عملکرد مدل پایدار بوده.")
    else:
        n_red = (alerts_df["شدت"] == "قرمز").sum()
        n_yellow = (alerts_df["شدت"] == "زرد").sum()
        a1, a2, a3 = st.columns(3)
        a1.metric("کل هشدارها", len(alerts_df))
        a2.metric("هشدار بحرانی 🔴", int(n_red))
        a3.metric("هشدار هشدار‌آمیز 🟡", int(n_yellow))

        severity_filter = st.multiselect("فیلتر بر اساس شدت", ["قرمز", "زرد"], default=["قرمز", "زرد"])
        filtered_alerts = alerts_df[alerts_df["شدت"].isin(severity_filter)]

        for _, row in filtered_alerts.iterrows():
            badge_class = "badge-red" if row["شدت"] == "قرمز" else "badge-yellow"
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom:8px;">
                    <span class="badge {badge_class}">{row['شدت']}</span>
                    <b style="margin-right:8px;">{row['نوع هشدار']}</b>
                    <span style="color:rgba(255,255,255,0.5); font-size:0.85rem; float:left;">{row['تاریخ']}</span>
                    <div style="margin-top:6px; color:rgba(255,255,255,0.85);">{row['پیام']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# تب ۶ — فیلتر، جست‌وجو و خروجی داده
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🗂️ فیلتر، جست‌وجو و خروجی داده</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">کاوش در لاگ پیش‌بینی‌های شبیه‌سازی‌شدهٔ تولید و دریافت خروجی</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1.2, 1, 1.4])
    with fc1:
        class_filter = st.multiselect(
            "فیلتر بر اساس کلاس پیش‌بینی‌شده",
            options=CLASS_IDS, format_func=lambda c: LABEL_NAMES[c],
            default=CLASS_IDS,
        )
    with fc2:
        conf_range = st.slider("بازهٔ اطمینان", 0.0, 1.0, (0.0, 1.0), 0.05)
    with fc3:
        search_term = st.text_input("🔎 جست‌وجو در متن", "")

    filtered = prod_log[
        prod_log["pred_label"].isin(class_filter)
        & prod_log["confidence"].between(*conf_range)
    ]
    if search_term.strip():
        filtered = filtered[filtered["text"].str.contains(search_term, case=False, na=False)]

    st.caption(f"🔢 تعداد نتایج: **{len(filtered):,}** از {len(prod_log):,} رکورد")

    display_df = filtered.copy()
    display_df["برچسب واقعی"] = display_df["label"].map(LABEL_NAMES)
    display_df["کلاس پیش‌بینی‌شده"] = display_df["pred_label"].map(LABEL_NAMES)
    display_df["اطمینان (%)"] = (display_df["confidence"] * 100).round(1)
    display_df = display_df.rename(columns={"timestamp": "زمان", "text": "متن"})
    st.dataframe(
        display_df[["زمان", "متن", "برچسب واقعی", "کلاس پیش‌بینی‌شده", "اطمینان (%)"]]
        .sort_values("زمان", ascending=False),
        use_container_width=True, hide_index=True, height=420,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    exp1, exp2 = st.columns(2)
    with exp1:
        csv_bytes = filtered.drop(columns=["token_len"]).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ دانلود CSV کامل (داده فیلترشده)",
            data=csv_bytes,
            file_name="crime_model_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        report_lines = [
            "گزارش خلاصهٔ مانیتورینگ مدل",
            "================================",
            f"مدل فعال: {model_name}",
            f"تاریخ گزارش: {datetime(2026, 8, 30).strftime('%Y-%m-%d')}",
            "",
            "-- معیارهای عملکرد (مجموعه تست) --",
            f"Accuracy: {acc*100:.2f}%",
            f"Precision: {prec*100:.2f}%",
            f"Recall: {rec*100:.2f}%",
            f"F1-Score: {f1*100:.2f}%",
            "",
            "-- وضعیت داده تولید --",
            f"تعداد رکوردهای ثبت‌شده: {len(prod_log)}",
            f"میانگین اطمینان: {prod_log['confidence'].mean()*100:.2f}%",
            f"میانگین طول متن (تولید): {prod_log['token_len'].mean():.2f} توکن",
            f"میانگین طول متن (مرجع): {ref_len.mean():.2f} توکن",
            f"آماره KS انحراف داده: {ks_stat:.3f} (p-value={ks_p:.4f})",
            "",
            "-- خطا --",
            f"تعداد نمونه با اطمینان پایین (<0.6): {(prod_log['confidence'] < 0.6).sum()}",
        ]
        report_text = "\n".join(report_lines)
        st.download_button(
            "⬇️ دانلود گزارش خلاصه (TXT)",
            data=report_text.encode("utf-8-sig"),
            file_name="monitoring_summary_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# تب ۷ — تفسیرپذیری (Explainability)
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">💡 تفسیرپذیری پیش‌بینی (Explainability)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">با حذف موقت هر کلمه (Occlusion)، تأثیر آن بر تصمیم مدل اندازه‌گیری و نمایش داده می‌شود</div>', unsafe_allow_html=True)

    mode = st.radio("منبع متن:", ["انتخاب از نمونه‌های تست", "نوشتن متن دلخواه"], horizontal=True)
    if mode == "انتخاب از نمونه‌های تست":
        sample_idx = st.selectbox(
            "یک نمونه انتخاب کنید",
            options=list(range(len(test_eval))),
            format_func=lambda i: test_eval.iloc[i]["text"][:70] + "...",
        )
        input_text = test_eval.iloc[sample_idx]["text"]
        true_label = test_eval.iloc[sample_idx]["label"]
    else:
        input_text = st.text_area("متن را وارد کنید (به انگلیسی، مطابق داده آموزشی):",
                                   "A masked man broke into a downtown bank branch late at night and stole a large amount of cash.")
        true_label = None

    if st.button("🔮 اجرای تحلیل تفسیرپذیری", type="primary"):
        model = models[model_name]
        base_ids = vectorize_texts(vectorizer, [input_text])[0]
        base_probs = model.predict(base_ids[None, :], verbose=0)[0]
        pred_class = int(np.argmax(base_probs))
        base_conf = float(base_probs[pred_class])

        words = input_text.strip().split()
        n_show = min(len(words), SEQ_LEN)

        importances = []
        for i in range(n_show):
            modified = base_ids.copy()
            modified[i] = 0  # حذف موقت توکن (جایگزینی با padding)
            probs_mod = model.predict(modified[None, :], verbose=0)[0]
            drop = base_conf - probs_mod[pred_class]
            importances.append(drop)
        importances = np.array(importances)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("کلاس پیش‌بینی‌شده", LABEL_NAMES[pred_class])
        r2.metric("اطمینان مدل", f"{base_conf*100:.1f}%")
        if true_label is not None:
            r3.metric("برچسب واقعی", LABEL_NAMES[int(true_label)])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖍️ برجسته‌سازی کلمات مؤثر در تصمیم مدل</div>', unsafe_allow_html=True)
        st.caption("رنگ پررنگ‌تر = تأثیر بیشتر کلمه در افزایش اطمینان مدل به کلاس پیش‌بینی‌شده")

        max_imp = max(importances.max(), 1e-6)
        html_words = []
        for w, imp in zip(words[:n_show], importances):
            norm = max(imp / max_imp, 0)
            alpha = 0.15 + 0.75 * norm
            color = LABEL_COLORS[pred_class]
            html_words.append(
                f'<span style="background:{color}{int(alpha*255):02x}; padding:3px 7px; '
                f'border-radius:6px; margin:2px; display:inline-block;">{w}</span>'
            )
        st.markdown(
            f'<div dir="ltr" style="text-align:left; line-height:2.4; font-size:1.05rem;">{" ".join(html_words)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 امتیاز اهمیت هر کلمه</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({"کلمه": words[:n_show], "امتیاز اهمیت": importances})
        imp_df = imp_df.sort_values("امتیاز اهمیت", ascending=True)
        fig_imp = px.bar(
            imp_df, x="امتیاز اهمیت", y="کلمه", orientation="h",
            template=PLOTLY_TEMPLATE, color="امتیاز اهمیت",
            color_continuous_scale="Purples",
        )
        fig_imp.update_layout(height=max(320, 24 * n_show), margin=dict(l=10, r=10, t=10, b=10),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 توزیع احتمال روی همهٔ کلاس‌ها</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "کلاس": [LABEL_NAMES[c] for c in CLASS_IDS],
            "احتمال": base_probs,
        })
        fig_p = px.bar(
            prob_df, x="کلاس", y="احتمال", color="کلاس",
            color_discrete_map={LABEL_NAMES[c]: LABEL_COLORS[c] for c in CLASS_IDS},
            template=PLOTLY_TEMPLATE,
        )
        fig_p.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; color:rgba(255,255,255,0.35); font-size:0.8rem; margin-top:30px;">
        داشبورد مانیتورینگ مدل · ساخته‌شده با Streamlit، TensorFlow/Keras و Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
