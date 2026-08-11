import re
import string
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Sentinel — Email Spam Detector",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme / CSS
# --------------------------------------------------------------------------
PRIMARY = "#5B21B6"      # deep violet
PRIMARY_2 = "#7C3AED"
ACCENT = "#F59E0B"       # amber accent
SPAM_RED = "#EF4444"
HAM_GREEN = "#10B981"
INK = "#1E1B2E"
PAPER = "#FAF9FC"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {PAPER};
    }}

    h1, h2, h3, .app-title {{
        font-family: 'Sora', sans-serif;
    }}

    /* Hero header */
    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_2} 55%, #C026D3 100%);
        border-radius: 20px;
        padding: 38px 42px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px -12px rgba(91,33,182,0.55);
        position: relative;
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        right: -60px; top: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .hero h1 {{
        font-size: 34px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }}
    .hero p {{
        font-size: 15.5px;
        opacity: 0.92;
        margin: 0;
        max-width: 640px;
    }}

    /* Pill tags */
    .pill {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 6px 3px 0;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.35);
        color: white;
    }}

    /* Metric / info cards */
    .metric-card {{
        background: white;
        border-radius: 16px;
        padding: 20px 22px;
        border: 1px solid #ECE9F5;
        box-shadow: 0 4px 14px -8px rgba(30,27,46,0.15);
        height: 100%;
    }}
    .metric-card .label {{
        font-size: 12.5px;
        font-weight: 600;
        color: #7C6FA0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .metric-card .value {{
        font-size: 28px;
        font-weight: 800;
        color: {INK};
        font-family: 'Sora', sans-serif;
    }}
    .metric-card .sub {{
        font-size: 12.5px;
        color: #94879C;
        margin-top: 4px;
    }}

    /* Result banners */
    .result-spam {{
        background: linear-gradient(120deg, #FEF2F2, #FEE2E2);
        border: 1.5px solid #FCA5A5;
        border-radius: 16px;
        padding: 22px 26px;
    }}
    .result-ham {{
        background: linear-gradient(120deg, #ECFDF5, #D1FAE5);
        border: 1.5px solid #6EE7B7;
        border-radius: 16px;
        padding: 22px 26px;
    }}
    .result-title {{
        font-family: 'Sora', sans-serif;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 4px;
    }}

    .word-chip {{
        display: inline-block;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 5px 3px 0;
    }}
    .chip-spam {{ background: #FEE2E2; color: #B91C1C; }}
    .chip-ham  {{ background: #D1FAE5; color: #047857; }}

    section[data-testid="stSidebar"] {{
        background: #1E1B2E;
    }}
    section[data-testid="stSidebar"] * {{
        color: #E9E6F5 !important;
    }}

    .footer-note {{
        text-align: center;
        color: #A79EC4;
        font-size: 12.5px;
        margin-top: 30px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    with open("metadata.json") as f:
        meta = json.load(f)
    return model, vectorizer, meta


@st.cache_data
def load_dataset():
    df = pd.read_csv("spam_dataset_clean.csv")
    return df


model, vectorizer, meta = load_artifacts()
df = load_dataset()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(message: str):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred = int(np.argmax(proba))
    return pred, proba, vec, cleaned


def top_contributing_words(vec, top_n=8):
    """Return top words in this message ranked by |coef * tfidf weight|, split by direction."""
    coefs = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    nz = vec.nonzero()[1]
    if len(nz) == 0:
        return [], []
    contributions = [(feature_names[i], coefs[i] * vec[0, i]) for i in nz]
    contributions.sort(key=lambda x: x[1], reverse=True)
    spammy = [c for c in contributions if c[1] > 0][:top_n]
    hammy = [c for c in contributions if c[1] < 0][:top_n]
    hammy.sort(key=lambda x: x[1])
    return spammy, hammy


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("### \U0001F6E1\uFE0F Spam Sentinel")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001F3E0 Overview", "\U0001F52C Try It Live", "\U0001F4CA Model Performance", "\U0001F4C8 Explore Dataset"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size:13px; line-height:1.7;">
    <b>Deployed model</b><br>{meta['deployed_model_name']}<br><br>
    <b>Vectorizer</b><br>TF-IDF (unigrams + bigrams, 5000 features)<br><br>
    <b>Training data</b><br>{meta['dataset_shape'][0]:,} labeled messages
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Sadia Zahid \u2022 Streamlit + scikit-learn")


# --------------------------------------------------------------------------
# Overview page
# --------------------------------------------------------------------------
if page.endswith("Overview"):
    st.markdown(
        f"""
        <div class="hero">
            <h1>\U0001F6E1\uFE0F Spam Sentinel</h1>
            <p>An ML-powered dashboard that classifies emails and messages as <b>Spam</b> or <b>Ham</b> in real time,
            using TF-IDF text features and a Logistic Regression classifier trained on {meta['dataset_shape'][0]:,} labeled messages.</p>
            <div style="margin-top:16px;">
                <span class="pill">TF-IDF Vectorization</span>
                <span class="pill">Logistic Regression</span>
                <span class="pill">{meta['results'][meta['deployed_model_name']]['accuracy']*100:.1f}% Accuracy</span>
                <span class="pill">Live Prediction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r = meta["results"][meta["deployed_model_name"]]
    cols = st.columns(4)
    cards = [
        ("Accuracy", f"{r['accuracy']*100:.1f}%", "on held-out test set"),
        ("Precision", f"{r['precision']*100:.1f}%", "of flagged spam is real spam"),
        ("Recall", f"{r['recall']*100:.1f}%", "of real spam gets caught"),
        ("F1 Score", f"{r['f1']*100:.1f}%", "precision / recall balance"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(
            f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="sub">{sub}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("#### How it works")
        st.markdown(
            """
            1. **Clean** — lowercase text, strip URLs / emails / punctuation / numbers.
            2. **Vectorize** — convert cleaned text into TF-IDF weighted unigram & bigram features (top 5,000 terms).
            3. **Classify** — a Logistic Regression model scores the probability the message is spam.
            4. **Explain** — the dashboard surfaces which words pushed the prediction toward spam or ham.
            """
        )

    with c2:
        counts = meta["class_counts"]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Ham", "Spam"],
                    values=[counts.get("ham", 0), counts.get("spam", 0)],
                    hole=0.62,
                    marker=dict(colors=[HAM_GREEN, SPAM_RED]),
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(
            title="Training data composition",
            margin=dict(t=50, b=10, l=10, r=10),
            height=300,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "\U0001F4CC A ready-to-use vectorizer wasn't included with the uploaded model file, so this dashboard "
        "trains a matching TF-IDF + Logistic Regression pipeline on the public SMS Spam Collection dataset "
        "(5,574 labeled messages) to power live predictions. Swap in your own dataset via `train_pipeline.py` any time.",
        icon="\u2139\uFE0F",
    )


# --------------------------------------------------------------------------
# Try it live
# --------------------------------------------------------------------------
elif page.endswith("Try It Live"):
    st.markdown("## \U0001F52C Try It Live")
    st.caption("Paste an email or message below and see how the model scores it.")

    examples = {
        "— choose an example —": "",
        "Spam-ish example": "Congratulations! You have WON a $1,000,000 prize. Click here immediately to claim your reward before it expires!!!",
        "Ham-ish example": "Hi, please attend the meeting tomorrow at 10 AM. Bring the project report with you.",
    }
    choice = st.selectbox("Quick examples", list(examples.keys()))
    default_text = examples[choice]

    message = st.text_area(
        "Message text",
        value=default_text,
        height=160,
        placeholder="Type or paste an email / message here...",
    )

    run = st.button("Analyze message", type="primary", use_container_width=True)

    if run:
        if not message.strip():
            st.warning("Please enter a message to analyze.")
        else:
            pred, proba, vec, cleaned = predict(message)
            spam_prob = proba[1] * 100
            ham_prob = proba[0] * 100

            left, right = st.columns([1, 1])

            with left:
                if pred == 1:
                    st.markdown(
                        f"""<div class="result-spam">
                                <div class="result-title" style="color:{SPAM_RED};">\U0001F6A8 Spam detected</div>
                                <div style="color:#7F1D1D; font-size:14px;">The model is <b>{spam_prob:.1f}%</b> confident this is spam.</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="result-ham">
                                <div class="result-title" style="color:{HAM_GREEN};">\u2705 Looks like Ham</div>
                                <div style="color:#065F46; font-size:14px;">The model is <b>{ham_prob:.1f}%</b> confident this is a legitimate message.</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

                st.write("")
                fig = go.Figure(
                    go.Bar(
                        x=[ham_prob, spam_prob],
                        y=["Ham", "Spam"],
                        orientation="h",
                        marker_color=[HAM_GREEN, SPAM_RED],
                        text=[f"{ham_prob:.1f}%", f"{spam_prob:.1f}%"],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    xaxis=dict(range=[0, 105], title="Confidence (%)"),
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("**Cleaned text seen by the model**")
                st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

                spammy, hammy = top_contributing_words(vec)
                st.markdown("**Words pushing the decision**")
                if spammy:
                    st.markdown(
                        "".join(f'<span class="word-chip chip-spam">{w}</span>' for w, _ in spammy),
                        unsafe_allow_html=True,
                    )
                if hammy:
                    st.markdown(
                        "".join(f'<span class="word-chip chip-ham">{w}</span>' for w, _ in hammy),
                        unsafe_allow_html=True,
                    )
                if not spammy and not hammy:
                    st.caption("No recognized vocabulary terms found in this message.")


# --------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------
elif page.endswith("Model Performance"):
    st.markdown("## \U0001F4CA Model Performance")

    results_df = pd.DataFrame(meta["results"]).T.reset_index().rename(columns={"index": "Model"})
    results_df[["accuracy", "precision", "recall", "f1"]] = results_df[
        ["accuracy", "precision", "recall", "f1"]
    ] * 100

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown("#### Model comparison")
        fig = go.Figure()
        metrics = ["accuracy", "precision", "recall", "f1"]
        colors = [PRIMARY, PRIMARY_2, ACCENT, "#C026D3"]
        for metric, color in zip(metrics, colors):
            fig.add_trace(
                go.Bar(
                    name=metric.capitalize(),
                    x=results_df["Model"],
                    y=results_df[metric],
                    marker_color=color,
                )
            )
        fig.update_layout(
            barmode="group",
            height=380,
            yaxis=dict(title="Score (%)", range=[0, 105]),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Highest test accuracy: **{meta['best_model_name']}** \u2014 currently deployed for live prediction: **{meta['deployed_model_name']}** "
            "(chosen for calibrated probability scores)."
        )

    with c2:
        st.markdown("#### Confusion matrix")
        cm = np.array(meta["confusion_matrix"])
        labels = meta["labels"]
        fig = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale=[[0, "#F3F0FA"], [1, PRIMARY]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Evaluated on {meta['n_test']:,} held-out test messages ({meta['n_train']:,} used for training).")

    st.markdown("#### Metrics table")
    st.dataframe(
        results_df.style.format({"accuracy": "{:.2f}%", "precision": "{:.2f}%", "recall": "{:.2f}%", "f1": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------------------------------
# Explore dataset
# --------------------------------------------------------------------------
elif page.endswith("Explore Dataset"):
    st.markdown("## \U0001F4C8 Explore Dataset")

    c1, c2, c3 = st.columns(3)
    counts = meta["class_counts"]
    total = sum(counts.values())
    for col, (label, val, color) in zip(
        [c1, c2, c3],
        [
            ("Total messages", f"{total:,}", INK),
            ("Ham messages", f"{counts.get('ham', 0):,}", HAM_GREEN),
            ("Spam messages", f"{counts.get('spam', 0):,}", SPAM_RED),
        ],
    ):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
                <div class="value" style="color:{color};">{val}</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    fig = px.histogram(
        df,
        x="message_length",
        color="label",
        nbins=50,
        color_discrete_map={"ham": HAM_GREEN, "spam": SPAM_RED},
        barmode="overlay",
        opacity=0.75,
        title="Message length distribution: Spam vs Ham",
    )
    fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Label")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        df,
        x="label",
        y="message_length",
        color="label",
        color_discrete_map={"ham": HAM_GREEN, "spam": SPAM_RED},
        title="Message length by class (outliers visible)",
    )
    fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Sample messages")
    label_filter = st.multiselect("Filter by label", options=["ham", "spam"], default=["ham", "spam"])
    sample = df[df["label"].isin(label_filter)][["label", "text", "message_length"]].sample(
        min(15, len(df)), random_state=1
    )
    st.dataframe(sample, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer-note">Spam Sentinel \u2014 TF-IDF + Logistic Regression \u2022 built with Streamlit</div>',
    unsafe_allow_html=True,
)