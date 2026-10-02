"""Streamlit football analytics dashboard — FotMob-inspired dark UI."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    MIN_MINUTES,
    GOAL_REGRESSION_FEATURES,
    SCORING_CLASSIFICATION_FEATURES,
    SCORING_THRESHOLD,
    clean_dataset,
    load_source_dataset,
)

# ── Page config ──
st.set_page_config(page_title="Football Goal Prediction", page_icon="⚽", layout="wide")

# ── FotMob-inspired CSS ──
st.markdown("""
<style>
/* ─── Global dark theme overrides ─── */
.stApp {
    background-color: #0e1117;
    color: #e0e0e0;
}

/* ─── Metric cards like FotMob ─── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1d23 0%, #1e2229 100%);
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
div[data-testid="stMetric"] label {
    color: #8b8d93 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ─── Tab styling ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background-color: #1a1d23;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #8b8d93;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background-color: #00c853 !important;
    color: #0e1117 !important;
    border-radius: 8px;
}

/* ─── Cards / containers ─── */
div[data-testid="stExpander"] {
    background-color: #1a1d23;
    border: 1px solid #2a2d35;
    border-radius: 12px;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
    color: #0e1117;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 8px 32px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e676 0%, #69f0ae 100%);
    transform: translateY(-1px);
}

/* ─── Success/error boxes ─── */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: none;
}

/* ─── Stat card helper ─── */
.stat-card {
    background: linear-gradient(135deg, #1a1d23 0%, #1e2229 100%);
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: #8b8d93;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.stat-card.green .stat-value { color: #00c853; }
.stat-card.red .stat-value { color: #ff5252; }
.stat-card.blue .stat-value { color: #448aff; }
.stat-card.orange .stat-value { color: #ff9800; }

/* ─── Player header ─── */
.player-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1a2b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 16px;
}
.player-name {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}
.player-info {
    color: #8b8d93;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ─── Prediction result ─── */
.prediction-box {
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    margin: 16px 0;
}
.prediction-box.likely {
    background: linear-gradient(135deg, #1b3a1b 0%, #0a2e0a 100%);
    border: 2px solid #00c853;
}
.prediction-box.unlikely {
    background: linear-gradient(135deg, #3a1b1b 0%, #2e0a0a 100%);
    border: 2px solid #ff5252;
}
.prediction-box .pred-icon { font-size: 2.5rem; }
.prediction-box .pred-text {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 8px;
}
.prediction-box .pred-prob {
    font-size: 2.5rem;
    font-weight: 900;
    margin-top: 4px;
}
.prediction-box.likely .pred-prob { color: #00c853; }
.prediction-box.unlikely .pred-prob { color: #ff5252; }
.prediction-box .pred-sub {
    color: #8b8d93;
    font-size: 0.85rem;
    margin-top: 8px;
}

/* ─── Section headers ─── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    border-left: 3px solid #00c853;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* ─── Dataframe styling ─── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Form border */
div[data-testid="stForm"] {
    background-color: #1a1d23;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──
def stat_card(value, label, color=""):
    cls = f" {color}" if color else ""
    return f"""<div class="stat-card{cls}">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""


def player_header_html(name, squad, comp, pos):
    return f"""<div class="player-header">
        <div class="player-name">{name}</div>
        <div class="player-info">{squad} · {comp} · {pos}</div>
    </div>"""


def section_header(text):
    return f'<div class="section-header">{text}</div>'


# ── Data loading ──
@st.cache_data
def load_players() -> pd.DataFrame:
    cleaned = ROOT / "data" / "processed" / "players_cleaned.csv"
    return pd.read_csv(cleaned) if cleaned.exists() else clean_dataset(load_source_dataset())


@st.cache_data
def load_clusters() -> pd.DataFrame:
    path = ROOT / "tables" / "player_clusters.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_table(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


players = load_players()
metrics = load_table("tables/regression_metrics.csv")
class_metrics = load_table("tables/classification_metrics.csv")

# ── Title ──
st.markdown("# ⚽ Football Player Analytics")
st.markdown('<p style="color: #8b8d93; margin-top: -10px;">2025–2026 · Top 5 European Leagues · Goal Scoring Prediction</p>', unsafe_allow_html=True)

# Build player lookup
choices = players.sort_values(["Player", "Squad"])
lookup = {
    f"{row.Player} — {row.Squad} ({row.Comp})": idx
    for idx, row in choices.iterrows()
}

tabs = st.tabs(["📊 Overview", "🔍 Player Analysis", "🎯 Goal Prediction", "⚡ Will Score?", "👥 Clusters"])

# ━━━ Tab 1: Overview ━━━
with tabs[0]:
    st.markdown(section_header("Season Overview"), unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(stat_card(f"{len(players):,}", "Player-Seasons", "blue"), unsafe_allow_html=True)
    c2.markdown(stat_card(players.Comp.nunique(), "Leagues", "green"), unsafe_allow_html=True)
    c3.markdown(stat_card(f"{players.Gls.mean():.1f}", "Avg Goals", "orange"), unsafe_allow_html=True)
    c4.markdown(stat_card(f"{players.Ast.mean():.1f}", "Avg Assists"), unsafe_allow_html=True)
    c5.markdown(stat_card(f"{players.Age.mean():.1f}", "Avg Age"), unsafe_allow_html=True)

    left, right = st.columns(2)
    for col, title, target in [
        (left, "Goals vs Shots", "plots/eda_goals_vs_shots.png"),
        (right, "Goals vs Shots on Target", "plots/eda_goals_vs_sot.png"),
    ]:
        col.markdown(section_header(title), unsafe_allow_html=True)
        image = ROOT / target
        if image.exists():
            col.image(str(image), use_container_width=True)

    st.markdown(section_header("Players by League"), unsafe_allow_html=True)
    st.bar_chart(players.Comp.value_counts())

# ━━━ Tab 2: Player Analysis ━━━
with tabs[1]:
    selected = st.selectbox("Select player-season", list(lookup), key="player_tab")
    player = players.loc[lookup[selected]]
    gpm = player.Gls / max(player.MP, 1)

    st.markdown(player_header_html(player.Player, player.Squad, player.Comp, player.Pos), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card(int(player.Age) if pd.notna(player.Age) else "?", "Age"), unsafe_allow_html=True)
    c2.markdown(stat_card(int(player.Min), "Minutes"), unsafe_allow_html=True)
    c3.markdown(stat_card(int(player.MP), "Matches"), unsafe_allow_html=True)
    c4.markdown(stat_card(f"{gpm:.3f}", "Goals/Match", "green" if gpm >= SCORING_THRESHOLD else "red"), unsafe_allow_html=True)

    if gpm >= SCORING_THRESHOLD:
        st.success(f"✅ **Likely to Score** — Averages a goal every {1/gpm:.0f} matches" if gpm > 0 else "")
    else:
        st.info(f"📊 Scores at a rate of {gpm:.3f} goals/match")

    st.markdown(section_header("Season Statistics"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Goals", int(player.Gls), "green"), ("Assists", int(player.Ast), "blue"),
        ("Shots", int(player.Sh), ""), ("Shots on Target", int(player.SoT), ""),
    ]
    for col, (label, val, clr) in zip([c1, c2, c3, c4], stats):
        col.markdown(stat_card(val, label, clr), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats2 = [
        ("Crosses", int(player.Crs), ""), ("Tackles Won", int(player.TklW), ""),
        ("Interceptions", int(player.Int), ""), ("Fouls Drawn", int(player.Fld), ""),
    ]
    for col, (label, val, clr) in zip([c1, c2, c3, c4], stats2):
        col.markdown(stat_card(val, label, clr), unsafe_allow_html=True)

# ━━━ Tab 3: Goal Prediction (Linear Regression) ━━━
with tabs[2]:
    st.markdown(section_header("Predict Season Goal Count"), unsafe_allow_html=True)
    st.caption("Linear Regression model trained on 1,600+ player-seasons with 450+ minutes")

    model_path = ROOT / "models" / "regression_linear.joblib"
    if not model_path.exists():
        st.warning("Run `python -m src.run_analysis` to train the model.")
    else:
        # Ronaldo's approximate stats (2024-25 Al Nassr, not in dataset)
        RONALDO_STATS = {
            "Age": 39.0, "Min": 2880.0, "Starts": 32.0, "Sh": 120.0,
            "Ast": 3.0, "Crs": 15.0, "TklW": 5.0, "Int": 4.0,
            "Fld": 45.0, "Fls": 30.0, "Off": 18.0, "CrdY": 4.0, "CrdR": 0.0,
        }

        input_mode = st.radio(
            "Input mode",
            ["Select a player", "Manual input", "🧪 Test with Ronaldo's data"],
            horizontal=True, key="reg_input_mode",
        )

        if input_mode == "Select a player":
            reg_player_sel = st.selectbox(
                "Choose player", list(lookup), key="reg_player_sel"
            )
            source = players.loc[lookup[reg_player_sel]]
            defaults = {f: float(source[f]) for f in GOAL_REGRESSION_FEATURES}
        elif input_mode == "🧪 Test with Ronaldo's data":
            st.info("📋 **Cristiano Ronaldo** — Al Nassr (estimated 2024-25 stats, not in dataset)")
            defaults = RONALDO_STATS.copy()
        else:
            defaults = {f: float(players[f].median()) for f in GOAL_REGRESSION_FEATURES}

        # Display current values in stat cards
        st.markdown(section_header("Input Features"), unsafe_allow_html=True)
        cols = st.columns(4)
        feature_labels = {
            "Age": "Age", "Min": "Minutes", "Starts": "Starts", "Sh": "Shots",
            "Ast": "Assists", "Crs": "Crosses", "TklW": "Tackles Won", "Int": "Interceptions",
            "Fld": "Fouls Drawn", "Fls": "Fouls", "Off": "Offsides", "CrdY": "Yellow Cards",
            "CrdR": "Red Cards",
        }

        entered = {}
        for i, feature in enumerate(GOAL_REGRESSION_FEATURES):
            with cols[i % 4]:
                entered[feature] = st.number_input(
                    feature_labels.get(feature, feature),
                    min_value=0.0 if feature != "Age" else 14.0,
                    value=defaults[feature],
                    step=1.0,
                    key=f"reg_live_{feature}_{input_mode}_{reg_player_sel if input_mode == 'Select a player' else 'x'}",
                )

        if st.button("⚽ Predict Goals", key="predict_goals_btn"):
            model = joblib.load(model_path)
            prediction = max(0.0, float(model.predict(pd.DataFrame([entered]))[0]))

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                color = "green" if prediction >= 5 else "orange" if prediction >= 2 else "red"
                st.markdown(f"""
                <div class="prediction-box {'likely' if prediction >= 5 else 'unlikely'}">
                    <div class="pred-icon">⚽</div>
                    <div class="pred-text">Predicted Season Goals</div>
                    <div class="pred-prob">{prediction:.1f}</div>
                    <div class="pred-sub">Linear Regression · {int(entered['Min'])} minutes input</div>
                </div>
                """, unsafe_allow_html=True)

            if entered["Min"] < MIN_MINUTES:
                st.warning("⚠️ Input is below the 450-minute training threshold — prediction may be unreliable.")

        # Model performance
        if not metrics.empty:
            st.markdown(section_header("Model Performance"), unsafe_allow_html=True)
            m = metrics.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.markdown(stat_card(f"{m.test_rmse:.2f}", "Test RMSE", "orange"), unsafe_allow_html=True)
            c2.markdown(stat_card(f"{m.test_mae:.2f}", "Test MAE", "blue"), unsafe_allow_html=True)
            c3.markdown(stat_card(f"{m.test_r2:.3f}", "Test R²", "green"), unsafe_allow_html=True)

        left, right = st.columns(2)
        for col, img, cap in [
            (left, "regression_actual_vs_predicted.png", "Actual vs Predicted"),
            (right, "regression_residuals.png", "Residuals"),
        ]:
            p = ROOT / "plots" / img
            if p.exists():
                col.image(str(p), caption=cap, use_container_width=True)

# ━━━ Tab 4: Will Score? ━━━
with tabs[3]:
    st.markdown(section_header("Next Match Scoring Prediction"), unsafe_allow_html=True)
    st.caption("Binary classification: Will this player score in the next match?")

    model_files = {
        "Logistic Regression": "models/classification_logistic.joblib",
        "Support Vector Machine": "models/classification_support.joblib",
    }
    available = [name for name, path in model_files.items() if (ROOT / path).exists()]

    if not available:
        st.warning("Run `python -m src.run_analysis` to train the models.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            score_player_sel = st.selectbox("Select a player", list(lookup), key="scoring_tab")
        with col2:
            selected_model_name = st.selectbox("Model", available, key="scoring_model")

        player_data = players.loc[lookup[score_player_sel]]
        st.markdown(
            player_header_html(player_data.Player, player_data.Squad, player_data.Comp, player_data.Pos),
            unsafe_allow_html=True,
        )

        # Show player's key stats
        c1, c2, c3, c4, c5 = st.columns(5)
        gpm_val = player_data.Gls / max(player_data.MP, 1)
        c1.markdown(stat_card(int(player_data.Gls), "Goals", "green"), unsafe_allow_html=True)
        c2.markdown(stat_card(int(player_data.Sh), "Shots"), unsafe_allow_html=True)
        c3.markdown(stat_card(int(player_data.SoT), "On Target"), unsafe_allow_html=True)
        c4.markdown(stat_card(int(player_data.Min), "Minutes"), unsafe_allow_html=True)
        c5.markdown(stat_card(f"{gpm_val:.2f}", "Goals/Match", "green" if gpm_val >= SCORING_THRESHOLD else "red"), unsafe_allow_html=True)

        if st.button("⚡ Predict Scoring Likelihood", key="predict_score_btn", type="primary"):
            model = joblib.load(ROOT / model_files[selected_model_name])
            features = {}
            for feat in SCORING_CLASSIFICATION_FEATURES:
                val = player_data.get(feat, np.nan)
                features[feat] = float(val) if pd.notna(val) else np.nan
            input_df = pd.DataFrame([features])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box likely">
                        <div class="pred-icon">⚽</div>
                        <div class="pred-text">LIKELY TO SCORE</div>
                        <div class="pred-prob">{probability[1]*100:.0f}%</div>
                        <div class="pred-sub">{selected_model_name} · Based on per-90 statistics</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box unlikely">
                        <div class="pred-icon">🛡️</div>
                        <div class="pred-text">UNLIKELY TO SCORE</div>
                        <div class="pred-prob">{probability[1]*100:.0f}%</div>
                        <div class="pred-sub">{selected_model_name} · Scoring probability shown above</div>
                    </div>
                    """, unsafe_allow_html=True)

            if player_data.Min < MIN_MINUTES:
                st.warning("⚠️ This player has fewer than 450 minutes — prediction may be unreliable.")

        # Model comparison
        if not class_metrics.empty:
            st.markdown(section_header("Model Comparison"), unsafe_allow_html=True)
            for _, r in class_metrics.iterrows():
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(stat_card(r.model.split()[0], "Model"), unsafe_allow_html=True)
                c2.markdown(stat_card(f"{r.test_accuracy:.1%}", "Accuracy", "blue"), unsafe_allow_html=True)
                c3.markdown(stat_card(f"{r.test_f1:.3f}", "F1 Score", "green"), unsafe_allow_html=True)
                c4.markdown(stat_card(f"{r.test_roc_auc:.3f}", "AUC", "orange"), unsafe_allow_html=True)

        st.markdown(section_header("Confusion Matrices"), unsafe_allow_html=True)
        left, right = st.columns(2)
        for col, tag, name in [
            (left, "logistic", "Logistic Regression"),
            (right, "support", "SVM"),
        ]:
            cm_path = ROOT / "plots" / f"confusion_matrix_{tag}.png"
            if cm_path.exists():
                col.image(str(cm_path), caption=name, use_container_width=True)

        st.markdown(section_header("ROC Curves"), unsafe_allow_html=True)
        left, right = st.columns(2)
        for col, tag, name in [
            (left, "logistic", "Logistic Regression"),
            (right, "support", "SVM"),
        ]:
            roc_path = ROOT / "plots" / f"roc_curve_{tag}.png"
            if roc_path.exists():
                col.image(str(roc_path), caption=name, use_container_width=True)

# ━━━ Tab 5: Clusters ━━━
with tabs[4]:
    st.markdown(section_header("Player Clusters (K-Means)"), unsafe_allow_html=True)
    clusters = load_clusters()
    if clusters.empty:
        st.warning("Run `python -m src.run_analysis` to discover player clusters.")
    else:
        cluster_choices = {
            f"{row.Player} — {row.Squad} ({row.Comp})": idx
            for idx, row in clusters.iterrows()
        }
        clustered_player = st.selectbox("Choose a player", list(cluster_choices), key="cluster_tab")
        record = clusters.loc[cluster_choices[clustered_player]]

        st.markdown(
            player_header_html(record.Player, record.Squad, record.Comp, record.Pos),
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.markdown(stat_card(int(record.cluster), "Cluster", "green"), unsafe_allow_html=True)
        c2.markdown(stat_card(f"{record.PC1:.2f}", "PC1", "blue"), unsafe_allow_html=True)
        c3.markdown(stat_card(f"{record.PC2:.2f}", "PC2", "orange"), unsafe_allow_html=True)

        # Show archetype
        interpretations = load_table("tables/cluster_interpretations.csv")
        if not interpretations.empty and "cluster" in interpretations.columns:
            match = interpretations.loc[interpretations.cluster == int(record.cluster)]
            if not match.empty:
                st.success(f"**Archetype:** {match.iloc[0].archetype}")

        image = ROOT / "plots" / "player_clusters_pca.png"
        if image.exists():
            st.image(str(image), use_container_width=True)

        criteria_image = ROOT / "plots" / "cluster_model_selection.png"
        if criteria_image.exists():
            st.image(str(criteria_image), use_container_width=True)

        profile = load_table("tables/cluster_profiles.csv")
        if not profile.empty and "cluster" in profile.columns:
            profile = profile.set_index("cluster")
            if int(record.cluster) in profile.index:
                st.markdown(section_header("Cluster Profile"), unsafe_allow_html=True)
                st.write(profile.loc[[int(record.cluster)]].T.rename(
                    columns={int(record.cluster): "Cluster Mean"}
                ))

        st.markdown(section_header("Similar Players"), unsafe_allow_html=True)
        peers = clusters.loc[
            clusters.cluster == record.cluster,
            ["Player", "Squad", "Comp", "Pos", "Min"]
        ]
        peers = peers.loc[
            ~((peers.Player == record.Player) & (peers.Squad == record.Squad) & (peers.Comp == record.Comp))
        ].head(10)
        st.dataframe(peers, use_container_width=True, hide_index=True)
