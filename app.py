import os
import pandas as pd
import streamlit as st
from src.predict import predict
from src.train import MODELS_DIR

CLASSIFIERS = ["logistic_regression", "svm", "xgboost"]
BEST_CLASSIFIER_FILE = os.path.join(MODELS_DIR, "best_model.txt")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
METRICS_CSV = os.path.join(REPORTS_DIR, "metrics.csv")

CLF_LABELS = {
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine",
    "xgboost": "XGBoost",
}


def classifiers_loaded():
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f"{clf}.joblib"))
        for clf in CLASSIFIERS
    )


def best_performing_classifier():
    if os.path.exists(BEST_CLASSIFIER_FILE):
        with open(BEST_CLASSIFIER_FILE) as f:
            return f.read().strip()
    return CLASSIFIERS[0]


st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for key, default in {
    "last_verdict": None,
    "scanned_sms": None,
    "spam_certainty": None,
    "scoring_classifier": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@1,9..144,300;1,9..144,700&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

:root {
    --bg:        #02091a;
    --bg2:       #061122;
    --bg3:       #0a1b35;
    --border:    #132d52;
    --blue:      #3d7cf4;
    --blue-lt:   #6fa3ff;
    --blue-dim:  rgba(61,124,244,0.1);
    --blue-glow: 0 0 28px rgba(61,124,244,0.22);
    --red:       #f0384f;
    --red-dim:   rgba(240,56,79,0.07);
    --teal:      #00bf85;
    --teal-dim:  rgba(0,191,133,0.07);
    --text:      #ddeeff;
    --text-mid:  #7a9cc4;
    --text-dim:  #2e4d72;
    --f-display: 'Fraunces', serif;
    --f-mono:    'IBM Plex Mono', monospace;
    --f-body:    'Plus Jakarta Sans', sans-serif;
}

*, html, body, [class*="css"] {
    font-family: var(--f-body) !important;
    background-color: transparent;
    color: var(--text);
    box-sizing: border-box;
}

.stApp {
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(61,124,244,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,191,133,0.06) 0%, transparent 55%);
}

.block-container { padding: 0 2.5rem 3rem !important; max-width: 1380px !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ── Header ─────────────────────────────────────────── */

.ss-header {
    padding: 2.8rem 0 0;
    margin-bottom: 0;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.6rem;
}
.ss-wordmark {
    display: flex;
    align-items: baseline;
    gap: 0.15em;
    line-height: 1;
}
.ss-word-spam {
    font-family: var(--f-display) !important;
    font-style: italic;
    font-weight: 700;
    font-size: 3.4rem;
    color: var(--blue-lt);
    letter-spacing: -3px;
}
.ss-word-shield {
    font-family: var(--f-mono) !important;
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    padding-bottom: 8px;
    padding-left: 2px;
}
.ss-right {
    text-align: right;
    padding-bottom: 2px;
}
.ss-author {
    font-family: var(--f-body) !important;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-mid);
}
.ss-course {
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 4px;
}
.ss-online {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 7px;
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
}
.ss-pulse {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 5px var(--teal);
    animation: pulse 2.8s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Tabs ────────────────────────────────────────────── */

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    margin-top: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--f-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.9rem 1.8rem !important;
    transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue-lt) !important;
    border-bottom-color: var(--blue) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-mid) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2.2rem !important; }

/* ── Form controls ───────────────────────────────────── */

label, .stSelectbox label, .stTextArea label {
    font-family: var(--f-mono) !important;
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--f-body) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: var(--blue-glow) !important;
}

.stTextArea textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--f-body) !important;
    font-size: 0.975rem !important;
    line-height: 1.7 !important;
    resize: none !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    caret-color: var(--blue-lt) !important;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: var(--blue-glow) !important;
}
.stTextArea textarea::placeholder { color: var(--text-dim) !important; }

.stButton > button {
    width: 100% !important;
    background: var(--blue) !important;
    border: none !important;
    border-radius: 4px !important;
    color: #fff !important;
    font-family: var(--f-mono) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.24em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
    box-shadow: 0 2px 18px rgba(61,124,244,0.35) !important;
}
.stButton > button:hover {
    background: #5490f6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 28px rgba(61,124,244,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Pipeline panel (left-rule only, no box) ─────────── */

.pipeline {
    border-left: 2px solid rgba(61,124,244,0.4);
    padding-left: 1.4rem;
    margin-top: 0.25rem;
}
.pipeline-eyebrow {
    font-family: var(--f-mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 1.4rem;
}
.pipeline-step { margin-bottom: 1.1rem; }
.pipeline-step:last-child { margin-bottom: 0; }
.p-num {
    font-family: var(--f-mono) !important;
    font-size: 0.58rem;
    color: var(--blue);
    margin-bottom: 0.2rem;
}
.p-text {
    font-family: var(--f-body) !important;
    font-size: 0.83rem;
    color: var(--text-mid);
    line-height: 1.6;
    font-weight: 300;
}
.top-clf {
    margin-top: 2rem;
    padding: 0.5rem 0;
    border-top: 1px solid var(--border);
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.top-clf span { color: var(--blue-lt); font-weight: 700; }

/* ── Pre-scan annotation (not a box) ─────────────────── */

.prescan-note {
    margin-top: 2rem;
    padding-left: 1rem;
    border-left: 2px solid var(--border);
    font-family: var(--f-body) !important;
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.6;
    font-style: italic;
}
.prescan-note strong { color: var(--text-mid); font-style: normal; font-weight: 500; }

/* ── Verdict panel ───────────────────────────────────── */

.verdict-wrap {
    margin-top: 2.2rem;
    border-top: 1px solid var(--border);
    padding-top: 2rem;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

.verdict-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.4rem;
}
.verdict-classification-spam {
    font-family: var(--f-display) !important;
    font-style: italic;
    font-weight: 700;
    font-size: 4rem;
    color: var(--red);
    line-height: 1;
    letter-spacing: -3px;
}
.verdict-classification-ham {
    font-family: var(--f-display) !important;
    font-style: italic;
    font-weight: 700;
    font-size: 4rem;
    color: var(--teal);
    line-height: 1;
    letter-spacing: -3px;
}
.verdict-meta {
    padding-bottom: 4px;
    text-align: right;
}
.verdict-finding {
    font-family: var(--f-body) !important;
    font-size: 0.875rem;
    color: var(--text-mid);
    font-weight: 400;
}
.verdict-scored-by {
    font-family: var(--f-mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 5px;
}

.certainty-section { max-width: 420px; margin-bottom: 1.6rem; }
.certainty-header {
    display: flex;
    justify-content: space-between;
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
}
.certainty-pct { color: var(--text-mid); }
.certainty-track { height: 5px; background: var(--bg3); border-radius: 1px; overflow: hidden; }
.certainty-bar-spam {
    height: 100%;
    background: var(--red);
    border-radius: 1px;
    animation: barGrow 0.7s cubic-bezier(0.16,1,0.3,1);
}
.certainty-bar-ham {
    height: 100%;
    background: var(--teal);
    border-radius: 1px;
    animation: barGrow 0.7s cubic-bezier(0.16,1,0.3,1);
}
@keyframes barGrow { from{width:0 !important} }

.evidence-block {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 0 4px 4px 0;
    padding: 0.9rem 1.1rem;
}
.evidence-eyebrow {
    font-family: var(--f-mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.evidence-text {
    font-family: var(--f-body) !important;
    font-size: 0.875rem;
    color: var(--text-mid);
    line-height: 1.65;
    font-weight: 300;
    word-break: break-word;
}

/* ── Benchmark tab ───────────────────────────────────── */

.bench-eyebrow {
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 1rem;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 5px !important; overflow: hidden !important; }

.bench-note {
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-top: 0.6rem;
    line-height: 1.7;
}

.charts-label {
    font-family: var(--f-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 2rem 0 1rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.chart-tile {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 1rem;
}
.chart-tile-label {
    font-family: var(--f-mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}
.no-chart {
    font-size: 0.78rem;
    color: var(--text-dim);
    font-style: italic;
    padding: 1.2rem 0;
}

.stAlert { border-radius: 4px !important; font-family: var(--f-body) !important; }

/* ── Mobile responsive ───────────────────────────────── */

@media (max-width: 768px) {
    .block-container { padding: 0 1rem 2rem !important; }

    .ss-header {
        grid-template-columns: 1fr;
        gap: 0.8rem;
        padding: 1.6rem 0 1.2rem;
    }
    .ss-right { text-align: left; }
    .ss-word-spam { font-size: 2.4rem; }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.6rem !important;
        padding: 0.7rem 0.9rem !important;
        letter-spacing: 0.08em !important;
    }

    .verdict-classification-spam,
    .verdict-classification-ham { font-size: 2.8rem; }

    .verdict-header { flex-direction: column; align-items: flex-start; }
    .verdict-meta { text-align: left; }

    .certainty-section { max-width: 100%; }

    .pipeline { padding-left: 1rem; }
}

@media (max-width: 480px) {
    .ss-word-spam { font-size: 2rem; }
    .verdict-classification-spam,
    .verdict-classification-ham { font-size: 2.2rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.6rem 0.7rem !important; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class="ss-header">
    <div class="ss-wordmark">
        <span class="ss-word-spam">Spam</span>
        <span class="ss-word-shield">/Shield</span>
    </div>
    <div class="ss-right">
        <div class="ss-author">Favour Nnenna Ogbonnaya-John</div>
        <div class="ss-course">COMP11117 · MSc IT · University of the West of Scotland</div>
        <div class="ss-online"><span class="ss-pulse"></span> {'3 classifiers online' if classifiers_loaded() else 'classifiers offline'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not classifiers_loaded():
    st.error("No trained classifiers found. Run `python main.py` to train the models before using this tool.")
    st.stop()

top_classifier = best_performing_classifier()

tab_scanner, tab_benchmarks = st.tabs(["Scan SMS", "Benchmark Results"])

# ── Scanner ────────────────────────────────────────────────────────────────────
with tab_scanner:
    col_compose, col_pipeline = st.columns([3, 2], gap="large")

    with col_compose:
        clf_default = CLASSIFIERS.index(top_classifier) if top_classifier in CLASSIFIERS else 0
        active_classifier = st.selectbox(
            "Detection Model",
            CLASSIFIERS,
            index=clf_default,
            format_func=lambda x: CLF_LABELS[x] + ("  ·  top F1" if x == top_classifier else ""),
        )
        sms_body = st.text_area(
            "SMS Body",
            height=172,
            placeholder="Paste the raw message. URLs, phone numbers and punctuation are stripped automatically before classification.",
        )
        scan_btn = st.button("Scan Message", type="primary", use_container_width=True)

    with col_pipeline:
        st.markdown(f"""
<div class="pipeline">
    <div class="pipeline-eyebrow">Classification pipeline</div>
    <div class="pipeline-step">
        <div class="p-num">01 · Normalise</div>
        <div class="p-text">Lowercase. Strip URLs, email addresses, phone numbers. Remove punctuation. Lemmatise to root forms.</div>
    </div>
    <div class="pipeline-step">
        <div class="p-num">02 · Vectorise</div>
        <div class="p-text">TF-IDF encodes the cleaned text as a 5,000-dimensional vector over unigrams and bigrams. High-frequency terms are down-weighted.</div>
    </div>
    <div class="pipeline-step">
        <div class="p-num">03 · Score</div>
        <div class="p-text">The selected classifier assigns a spam probability. SMOTE-balanced training corrects the 87:13 ham/spam skew.</div>
    </div>
    <div class="top-clf">Highest F1 · <span>{CLF_LABELS.get(top_classifier, top_classifier)}</span></div>
</div>
""", unsafe_allow_html=True)

    if scan_btn:
        if not sms_body.strip():
            st.session_state.last_verdict = None
        else:
            with st.spinner(""):
                verdict, spam_probability = predict(sms_body, active_classifier)
            st.session_state.last_verdict = verdict
            st.session_state.scanned_sms = sms_body
            st.session_state.spam_certainty = spam_probability if spam_probability is not None else 0.0
            st.session_state.scoring_classifier = active_classifier

    if st.session_state.last_verdict is not None:
        is_spam = st.session_state.last_verdict == "Spam"
        certainty = st.session_state.spam_certainty
        clf_used = CLF_LABELS.get(st.session_state.scoring_classifier, st.session_state.scoring_classifier)
        classification_cls = "verdict-classification-spam" if is_spam else "verdict-classification-ham"
        bar_cls = "certainty-bar-spam" if is_spam else "certainty-bar-ham"
        finding = (
            "Unsolicited commercial message — spam indicators present."
            if is_spam else
            "No spam indicators detected. Message appears legitimate."
        )
        preview = st.session_state.scanned_sms[:280].replace("<", "&lt;").replace(">", "&gt;")
        if len(st.session_state.scanned_sms) > 280:
            preview += "…"

        st.markdown(f"""
<div class="verdict-wrap">
    <div class="verdict-header">
        <div class="{classification_cls}">{st.session_state.last_verdict}</div>
        <div class="verdict-meta">
            <div class="verdict-finding">{finding}</div>
            <div class="verdict-scored-by">Scored by {clf_used}</div>
        </div>
    </div>
    <div class="certainty-section">
        <div class="certainty-header">
            <span>Classifier certainty</span>
            <span class="certainty-pct">{certainty:.1%}</span>
        </div>
        <div class="certainty-track">
            <div class="{bar_cls}" style="width:{certainty*100:.1f}%"></div>
        </div>
    </div>
    <div class="evidence-block">
        <div class="evidence-eyebrow">Scanned message</div>
        <div class="evidence-text">{preview}</div>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="prescan-note">
    Paste an SMS message above and click <strong>Scan Message</strong> to classify it.
    The verdict persists if you switch detection models.
</div>
""", unsafe_allow_html=True)

# ── Benchmarks ─────────────────────────────────────────────────────────────────
with tab_benchmarks:
    if os.path.exists(METRICS_CSV):
        st.markdown('<div class="bench-eyebrow">Held-out test set · 80/20 stratified split · 5,574 labelled messages</div>', unsafe_allow_html=True)

        metrics_df = pd.read_csv(METRICS_CSV)
        metrics_df.columns = [c.replace("_", " ").title() for c in metrics_df.columns]
        scored_cols = [c for c in metrics_df.columns if c != "Model"]
        top_row = metrics_df[scored_cols].idxmax()

        def highlight_top(col):
            return [
                "background-color:rgba(61,124,244,0.1); color:#6fa3ff; font-weight:600"
                if i == top_row[col.name] else ""
                for i in col.index
            ]

        st.dataframe(
            metrics_df.style
                .apply(highlight_top, subset=scored_cols)
                .format({c: "{:.4f}" for c in scored_cols}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<div class="bench-note">'
            'F1-score and ROC-AUC are primary metrics. '
            'Accuracy is unreliable on the 87:13 ham/spam imbalance — a model that always predicts ham achieves 87% accuracy without learning anything.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
<div class="prescan-note">
    No benchmark data. Run <strong>python main.py</strong> to train the classifiers and generate evaluation results.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="charts-label">Evaluation charts</div>', unsafe_allow_html=True)

    charts = {
        "Model Comparison": os.path.join(REPORTS_DIR, "model_comparison.png"),
        "ROC Curves": os.path.join(REPORTS_DIR, "roc_curves.png"),
        "Confusion Matrices": os.path.join(REPORTS_DIR, "confusion_matrices.png"),
    }

    # Intentionally asymmetric: wider chart gets more visual weight
    wide_col, mid_col, narrow_col = st.columns([5, 4, 4], gap="medium")
    col_map = [wide_col, mid_col, narrow_col]

    for col, (label, path) in zip(col_map, charts.items()):
        with col:
            st.markdown(f'<div class="chart-tile"><div class="chart-tile-label">{label}</div>', unsafe_allow_html=True)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.markdown('<div class="no-chart">Run main.py to generate.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
