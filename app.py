import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "loan_status"
LABELS = {0: "Good", 1: "Bad"}
NOTEBOOK_COMPARISON = [
    {
        "Algorithm Name": "Random Forest",
        "Accuracy": 87.218,
        "Precision": 82.814,
        "Recall": 77.066,
        "FSCORE": 79.367,
    },
    {
        "Algorithm Name": "XGBoost",
        "Accuracy": 86.650,
        "Precision": 80.046,
        "Recall": 81.555,
        "FSCORE": 80.755,
    },
    {
        "Algorithm Name": "Propose Hybrid LSTM",
        "Accuracy": 88.538,
        "Precision": 86.521,
        "Recall": 77.545,
        "FSCORE": 80.820,
    },
    {
        "Algorithm Name": "Extension Hybrid LSTM + Attention",
        "Accuracy": 89.873,
        "Precision": 85.753,
        "Recall": 83.506,
        "FSCORE": 84.551,
    },
]


MODEL_CANDIDATES = {
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        learning_rate=0.06,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=350,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
        n_estimators=250,
        learning_rate=0.06,
        max_depth=3,
    ),
    "Logistic Regression": LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
    ),
}


st.set_page_config(
    page_title="Extension Credit Score",
    page_icon="EC",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #18212f;
        --muted: #627084;
        --blue: #2563eb;
        --green: #0f9f6e;
        --amber: #d97706;
        --rose: #e11d48;
        --panel: #ffffff;
        --line: #d9e2ef;
    }

    .block-container {
        padding-top: 2rem;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
        background:
            linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(15, 159, 110, 0.09)),
            #ffffff;
        animation: fade-up 520ms ease-out;
    }

    .hero h1 {
        color: var(--ink);
        font-size: 34px;
        line-height: 1.1;
        margin: 0 0 8px;
        letter-spacing: 0;
    }

    .hero p {
        color: var(--muted);
        margin: 0;
        font-size: 15px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 10px;
    }

    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 15px 16px;
        position: relative;
        overflow: hidden;
        animation: fade-up 520ms ease-out;
    }

    .metric-card:before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: var(--accent);
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--ink);
        font-size: 25px;
        font-weight: 800;
        line-height: 1;
    }

    .result-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 18px;
        background: #ffffff;
        box-shadow: 0 10px 30px rgba(24, 33, 47, 0.08);
        animation: pulse-in 420ms ease-out;
    }

    .result-good {
        border-left: 7px solid var(--green);
    }

    .result-bad {
        border-left: 7px solid var(--rose);
    }

    .result-title {
        font-size: 13px;
        color: var(--muted);
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 850;
        color: var(--ink);
        line-height: 1.1;
    }

    .why-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 12px;
    }

    .why-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
        padding: 15px 16px;
        animation: fade-up 620ms ease-out;
    }

    .why-card strong {
        color: var(--ink);
    }

    .why-card span {
        display: block;
        color: var(--muted);
        margin-top: 5px;
        font-size: 13px;
    }

    .tab-banner {
        border-radius: 8px;
        padding: 16px 18px;
        margin: 6px 0 16px;
        border: 1px solid var(--line);
        animation: fade-up 520ms ease-out;
    }

    .tab-banner h3 {
        margin: 0 0 5px;
        color: var(--ink);
        font-size: 21px;
        line-height: 1.15;
    }

    .tab-banner p {
        margin: 0;
        color: var(--muted);
        font-size: 14px;
    }

    .banner-blue {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.13), rgba(14, 165, 233, 0.08)), #ffffff;
    }

    .banner-green {
        background: linear-gradient(135deg, rgba(15, 159, 110, 0.14), rgba(132, 204, 22, 0.08)), #ffffff;
    }

    .banner-amber {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.14), rgba(244, 63, 94, 0.08)), #ffffff;
    }

    .banner-slate {
        background: linear-gradient(135deg, rgba(71, 85, 105, 0.11), rgba(37, 99, 235, 0.07)), #ffffff;
    }

    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 16px;
    }

    .mini-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
        padding: 14px 15px;
        border-top: 4px solid var(--accent);
        box-shadow: 0 8px 20px rgba(24, 33, 47, 0.05);
        animation: fade-up 560ms ease-out;
    }

    .mini-card strong {
        color: var(--ink);
        display: block;
        font-size: 15px;
    }

    .mini-card span {
        color: var(--muted);
        display: block;
        margin-top: 5px;
        font-size: 13px;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 16px;
    }

    .pill {
        border-radius: 999px;
        padding: 7px 11px;
        background: #eef4ff;
        color: #1d4ed8;
        border: 1px solid #c7d7fe;
        font-size: 12px;
        font-weight: 750;
    }

    .model-chip {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-weight: 800;
        font-size: 12px;
        margin-left: 6px;
    }

    .compare-banner {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        margin: 8px 0 16px;
        background:
            linear-gradient(135deg, rgba(15, 159, 110, 0.13), rgba(37, 99, 235, 0.10)),
            #ffffff;
        animation: fade-up 520ms ease-out;
    }

    .compare-banner strong {
        display: block;
        color: var(--ink);
        font-size: 20px;
        margin-bottom: 4px;
    }

    .compare-banner span {
        color: var(--muted);
        font-size: 14px;
    }

    .leaderboard {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        margin-top: 10px;
    }

    .leader-row {
        display: grid;
        grid-template-columns: 34px 1fr 86px;
        gap: 10px;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid #edf2f7;
    }

    .leader-row:last-child {
        border-bottom: 0;
    }

    .leader-rank {
        height: 26px;
        width: 26px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        background: var(--blue);
        font-weight: 800;
        font-size: 12px;
    }

    .leader-name {
        color: var(--ink);
        font-weight: 750;
    }

    .leader-score {
        color: var(--green);
        font-weight: 850;
        text-align: right;
    }

    @keyframes fade-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-in {
        from { opacity: 0; transform: scale(0.985); }
        to { opacity: 1; transform: scale(1); }
    }

    @media (max-width: 900px) {
        .metric-grid,
        .why-grid,
        .mini-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 640px) {
        .metric-grid,
        .why-grid,
        .mini-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class CreditScorePipeline:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model
        self.feature_columns = None
        self.pipeline = None

    def fit(self, features, target):
        x = features.copy()
        y = target.astype(int)
        self.feature_columns = list(x.columns)

        categorical_columns = x.select_dtypes(include=["object", "string"]).columns.tolist()
        numeric_columns = [
            column for column in self.feature_columns if column not in categorical_columns
        ]

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                ),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", self.model),
            ]
        )
        self.pipeline.fit(x, y)
        return self

    def predict(self, features):
        x = features.reindex(columns=self.feature_columns)
        return self.pipeline.predict(x)

    def predict_proba(self, features):
        if hasattr(self.pipeline, "predict_proba"):
            x = features.reindex(columns=self.feature_columns)
            return self.pipeline.predict_proba(x)
        return None


@st.cache_data(show_spinner=False)
def load_packaged_dataset():
    try:
        return pd.read_csv("Dataset/credit_risk_dataset.csv")
    except FileNotFoundError:
        return None


def train_pipeline(dataset):
    dataset = dataset.dropna(subset=[TARGET_COLUMN])
    features = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if target.nunique() > 1 else None,
    )

    results = []
    trained_models = {}
    best_name = None
    best_accuracy = -1

    for model_name, model in MODEL_CANDIDATES.items():
        candidate = CreditScorePipeline(model_name, clone(model)).fit(x_train, y_train)
        predictions = candidate.predict(x_test)
        metrics = {
            "Algorithm Name": model_name,
            "Accuracy": round(accuracy_score(y_test.astype(int), predictions) * 100, 3),
            "Precision": round(
                precision_score(
                    y_test.astype(int),
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
                3,
            ),
            "Recall": round(
                recall_score(
                    y_test.astype(int),
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
                3,
            ),
            "FSCORE": round(
                f1_score(
                    y_test.astype(int),
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
                3,
            ),
        }
        results.append(metrics)
        trained_models[model_name] = candidate
        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_name = model_name

    pipeline = trained_models[best_name]
    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test.astype(int),
        predictions,
        target_names=["Good Score", "Bad Score"],
        output_dict=True,
        zero_division=0,
    )

    return pipeline, best_accuracy / 100, report, x_test, pd.DataFrame(results)


@st.cache_resource(show_spinner=False)
def train_pipeline_cached(dataset_bytes):
    dataset = pd.read_csv(io.BytesIO(dataset_bytes)).dropna(how="all")
    return train_pipeline(dataset)


def build_comparison_frame(model_results):
    model_results = model_results.copy()
    model_results["Source"] = "Auto-tested in app"
    notebook_results = pd.DataFrame(NOTEBOOK_COMPARISON)
    notebook_results["Source"] = "Notebook reference"
    comparison = pd.concat([notebook_results, model_results], ignore_index=True)
    comparison["Display Name"] = comparison.apply(
        lambda row: f"{row['Algorithm Name']} ({row['Source']})",
        axis=1,
    )
    return comparison


def build_input_form(reference_frame):
    values = {}

    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, column in enumerate(reference_frame.columns):
            series = reference_frame[column]
            with columns[index % 2]:
                if is_numeric_dtype(series):
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    default = (
                        float(numeric_series.dropna().median())
                        if numeric_series.notna().any()
                        else 0.0
                    )
                    values[column] = st.number_input(column, value=default)
                else:
                    options = sorted(series.dropna().astype(str).unique())
                    if not options:
                        options = [""]
                    values[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("Predict Credit Score", type="primary")

    return submitted, pd.DataFrame([values])


st.markdown(
    """
    <div class="hero">
        <h1>Extension Credit Score</h1>
        <p>Train on the credit risk dataset, predict Good or Bad credit score, and compare model performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Training Dataset")
uploaded = st.sidebar.file_uploader(
    "Optional: replace training CSV",
    type=["csv"],
    help="Upload credit_risk_dataset.csv only if you want to retrain with a different training file.",
)

if uploaded is not None:
    dataset_bytes = uploaded.getvalue()
    dataset = pd.read_csv(io.BytesIO(dataset_bytes))
    training_source = "Uploaded training CSV"
else:
    try:
        with open("Dataset/credit_risk_dataset.csv", "rb") as dataset_file:
            dataset_bytes = dataset_file.read()
        dataset = pd.read_csv(io.BytesIO(dataset_bytes))
    except FileNotFoundError:
        dataset_bytes = None
        dataset = None
    training_source = "Dataset/credit_risk_dataset.csv"

if dataset is None:
    st.info(
        "Add the training CSV at `Dataset/credit_risk_dataset.csv`, or upload "
        "`credit_risk_dataset.csv` from the sidebar."
    )
    st.stop()

if TARGET_COLUMN not in dataset.columns:
    st.error(f"The dataset must include a `{TARGET_COLUMN}` column.")
    st.stop()

dataset = dataset.dropna(how="all")
if dataset_bytes is None:
    st.stop()

with st.spinner("Training and comparing models. Cached results are reused on refresh."):
    pipeline, accuracy, report, test_features, model_results = train_pipeline_cached(
        dataset_bytes
    )
feature_frame = dataset.drop(columns=[TARGET_COLUMN])
comparison_df = build_comparison_frame(model_results)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card" style="--accent: var(--blue);">
            <div class="metric-label">Rows</div>
            <div class="metric-value">{len(dataset):,}</div>
        </div>
        <div class="metric-card" style="--accent: var(--amber);">
            <div class="metric-label">Features</div>
            <div class="metric-value">{feature_frame.shape[1]:,}</div>
        </div>
        <div class="metric-card" style="--accent: var(--green);">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{accuracy * 100:.2f}%</div>
        </div>
        <div class="metric-card" style="--accent: var(--rose);">
            <div class="metric-label">Model</div>
            <div class="metric-value">{pipeline.model_name}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Training source: {training_source}")

tab_predict, tab_batch, tab_compare, tab_metrics, tab_data = st.tabs(
    [
        "Manual Prediction",
        "Test CSV Prediction",
        "Comparison",
        "Metrics",
        "Training Data",
    ]
)

with tab_predict:
    st.markdown(
        """
        <div class="tab-banner banner-blue">
            <h3>Manual Credit Score Prediction</h3>
            <p>Enter one applicant profile and let the selected model classify the credit score as Good or Bad.</p>
        </div>
        <div class="mini-grid">
            <div class="mini-card" style="--accent: var(--blue);">
                <strong>Single applicant</strong>
                <span>Use this tab when you want one instant prediction from manually entered values.</span>
            </div>
            <div class="mini-card" style="--accent: var(--green);">
                <strong>Active model</strong>
                <span>The prediction uses the current best model selected from the training CSV.</span>
            </div>
            <div class="mini-card" style="--accent: var(--amber);">
                <strong>Confidence view</strong>
                <span>The result includes a confidence bar when the selected model supports probabilities.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    submitted, input_frame = build_input_form(feature_frame)
    if submitted:
        prediction = int(pipeline.predict(input_frame)[0])
        probabilities = pipeline.predict_proba(input_frame)
        prediction_label = LABELS.get(prediction, prediction)
        result_class = "result-good" if prediction == 0 else "result-bad"
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-title">Predicted Credit Score</div>
                <div class="result-value">{prediction_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if probabilities is not None and probabilities.shape[1] == 2:
            st.progress(float(probabilities[0][prediction]))
            st.caption(f"Confidence: {probabilities[0][prediction] * 100:.2f}%")

with tab_batch:
    st.markdown(
        """
        <div class="tab-banner banner-green">
            <h3>Batch Test CSV Prediction</h3>
            <p>Upload a test dataset with the same feature columns and download predictions for every row.</p>
        </div>
        <div class="pill-row">
            <span class="pill">CSV input</span>
            <span class="pill">loan_status optional</span>
            <span class="pill">Download predictions</span>
            <span class="pill">Good / Bad summary</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    batch_file = st.file_uploader("Upload testData.csv", type=["csv"], key="batch")
    if batch_file is not None:
        batch_data = pd.read_csv(batch_file)
        missing_columns = [
            column for column in feature_frame.columns if column not in batch_data.columns
        ]
        if missing_columns:
            st.error(
                "Your test CSV is missing these required feature columns: "
                + ", ".join(missing_columns)
            )
            st.stop()
        predictions = pipeline.predict(batch_data)
        output = batch_data.copy()
        output["predicted_credit_score"] = [
            LABELS.get(int(value), str(value)) for value in predictions
        ]
        summary = output["predicted_credit_score"].value_counts().rename_axis("Prediction")
        st.markdown("#### Prediction Summary")
        summary_df = summary.reset_index(name="Count")
        colors = ["#0f9f6e" if label == "Good" else "#e11d48" for label in summary_df["Prediction"]]
        fig_batch, ax_batch = plt.subplots(figsize=(7.5, 3.5))
        fig_batch.patch.set_facecolor("white")
        ax_batch.set_facecolor("white")
        bars = ax_batch.bar(
            summary_df["Prediction"],
            summary_df["Count"],
            color=colors,
            edgecolor="white",
            linewidth=1.2,
        )
        ax_batch.bar_label(bars, padding=4, fontsize=10, color="#18212f")
        ax_batch.set_ylabel("Applicants", color="#18212f")
        ax_batch.grid(axis="y", color="#e7edf5", linewidth=1)
        ax_batch.spines[["top", "right", "left"]].set_visible(False)
        ax_batch.spines["bottom"].set_color("#d9e2ef")
        st.pyplot(fig_batch, use_container_width=True)
        plt.close(fig_batch)
        st.markdown("#### Prediction Output")
        st.dataframe(output, width="stretch")
        st.download_button(
            "Download Predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="credit_score_predictions.csv",
            mime="text/csv",
        )

with tab_compare:
    st.markdown(
        """
        <div class="tab-banner banner-amber">
            <h3>Algorithm Performance Comparison</h3>
            <p>Compare live auto-tested models against your notebook reference results using Accuracy, Precision, Recall, and FSCORE.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_live = model_results.sort_values("Accuracy", ascending=False).iloc[0]
    st.markdown(
        (
            '<div class="compare-banner">'
            f'<strong>Best live model: {top_live["Algorithm Name"]} - '
            f'{top_live["Accuracy"]:.3f}% accuracy</strong>'
            "<span>The app tests multiple algorithms on the active training CSV, "
            "then uses the best model for predictions. Training results are cached "
            "until the training CSV changes.</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Live Auto-Tested Models")
    live_plot_df = model_results.sort_values("Accuracy", ascending=False).set_index(
        "Algorithm Name"
    )
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "FSCORE"]
    metric_colors = ["#2563eb", "#0f9f6e", "#d97706", "#e11d48"]
    x = np.arange(len(live_plot_df.index))
    bar_width = 0.18
    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for offset, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
        positions = x + (offset - 1.5) * bar_width
        bars = ax.bar(
            positions,
            live_plot_df[metric],
            width=bar_width,
            label=metric,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        if metric == "Accuracy":
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color="#18212f")

    ax.set_ylim(70, 100)
    ax.set_ylabel("Performance (%)", color="#18212f", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(live_plot_df.index, rotation=18, ha="right", fontsize=10)
    ax.grid(axis="y", color="#e7edf5", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d9e2ef")
    ax.legend(
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        frameon=False,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("#### Accuracy Ranking")
        leaders = comparison_df.sort_values("Accuracy", ascending=False).head(5)
        rows = []
        for rank, (_, row) in enumerate(leaders.iterrows(), start=1):
            rows.append(
                (
                    '<div class="leader-row">'
                    f'<div class="leader-rank">{rank}</div>'
                    f'<div class="leader-name">{row["Algorithm Name"]}</div>'
                    f'<div class="leader-score">{row["Accuracy"]:.2f}%</div>'
                    "</div>"
                )
            )
        st.markdown(
            f'<div class="leaderboard">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown("#### Detailed Scores")
        display_cols = ["Algorithm Name", "Source", "Accuracy", "Precision", "Recall", "FSCORE"]
        st.dataframe(
            comparison_df[display_cols].sort_values("Accuracy", ascending=False),
            width="stretch",
        )

    with st.expander("Notebook reference graph"):
        notebook_df = pd.DataFrame(NOTEBOOK_COMPARISON)
        notebook_plot_df = notebook_df.sort_values(
            "Accuracy", ascending=False
        ).set_index("Algorithm Name")
        x_ref = np.arange(len(notebook_plot_df.index))
        fig_ref, ax_ref = plt.subplots(figsize=(13, 5.2))
        fig_ref.patch.set_facecolor("white")
        ax_ref.set_facecolor("white")

        for offset, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
            positions = x_ref + (offset - 1.5) * bar_width
            bars = ax_ref.bar(
                positions,
                notebook_plot_df[metric],
                width=bar_width,
                label=metric,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            if metric == "Accuracy":
                ax_ref.bar_label(
                    bars, fmt="%.2f", padding=3, fontsize=8, color="#18212f"
                )

        ax_ref.set_ylim(70, 100)
        ax_ref.set_ylabel("Notebook Score (%)", color="#18212f", fontsize=11)
        ax_ref.set_xticks(x_ref)
        ax_ref.set_xticklabels(
            notebook_plot_df.index, rotation=18, ha="right", fontsize=10
        )
        ax_ref.grid(axis="y", color="#e7edf5", linewidth=1)
        ax_ref.spines[["top", "right", "left"]].set_visible(False)
        ax_ref.spines["bottom"].set_color("#d9e2ef")
        ax_ref.legend(
            ncols=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.13),
            frameon=False,
        )
        st.pyplot(fig_ref, use_container_width=True)
        plt.close(fig_ref)

    best_accuracy = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
    notebook_best = max(row["Accuracy"] for row in NOTEBOOK_COMPARISON)
    lift = best_accuracy["Accuracy"] - notebook_best
    lift_text = (
        f"about {lift:.3f} points above the best notebook model"
        if lift >= 0
        else f"about {abs(lift):.3f} points below the best notebook model"
    )
    st.markdown(
        f"""
        <div class="why-grid">
            <div class="why-card">
                <strong>Automatic winner</strong>
                <span>{pipeline.model_name} is selected because it has the highest validation accuracy on the active training CSV.</span>
            </div>
            <div class="why-card">
                <strong>Current best score</strong>
                <span>{best_accuracy["Algorithm Name"]} reaches {best_accuracy["Accuracy"]:.3f}%, {lift_text}.</span>
            </div>
            <div class="why-card">
                <strong>Adapts to new data</strong>
                <span>If a new training CSV changes the winner, the app automatically uses that better model for predictions.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_metrics:
    st.markdown(
        f"""
        <div class="tab-banner banner-slate">
            <h3>Validation Metrics <span class="model-chip">{pipeline.model_name}</span></h3>
            <p>Detailed classification report for the model selected from the active training CSV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = pd.DataFrame(report).transpose()
    metric_overview = metrics.loc[["Good Score", "Bad Score"], ["precision", "recall", "f1-score"]]
    metric_plot = metric_overview.reset_index(names="Class").melt(
        id_vars="Class",
        var_name="Metric",
        value_name="Score",
    )
    metric_plot["Score"] = metric_plot["Score"] * 100
    classes = metric_overview.index.tolist()
    x_metrics = np.arange(len(classes))
    width_metrics = 0.24
    fig_metrics, ax_metrics = plt.subplots(figsize=(9, 4))
    fig_metrics.patch.set_facecolor("white")
    ax_metrics.set_facecolor("white")
    for offset, (metric_name, color) in enumerate(
        zip(["precision", "recall", "f1-score"], ["#2563eb", "#0f9f6e", "#d97706"])
    ):
        values = metric_overview[metric_name].values * 100
        positions = x_metrics + (offset - 1) * width_metrics
        bars = ax_metrics.bar(
            positions,
            values,
            width=width_metrics,
            label=metric_name.title(),
            color=color,
            edgecolor="white",
            linewidth=0.9,
        )
        ax_metrics.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax_metrics.set_ylim(0, 100)
    ax_metrics.set_ylabel("Score (%)", color="#18212f")
    ax_metrics.set_xticks(x_metrics)
    ax_metrics.set_xticklabels(classes)
    ax_metrics.grid(axis="y", color="#e7edf5", linewidth=1)
    ax_metrics.spines[["top", "right", "left"]].set_visible(False)
    ax_metrics.spines["bottom"].set_color("#d9e2ef")
    ax_metrics.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    st.pyplot(fig_metrics, use_container_width=True)
    plt.close(fig_metrics)
    st.dataframe(metrics, width="stretch")

with tab_data:
    st.markdown(
        """
        <div class="tab-banner banner-blue">
            <h3>Training Data Preview</h3>
            <p>Inspect the first rows from the active training CSV used for model selection and prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_counts = dataset[TARGET_COLUMN].value_counts().sort_index()
    good_count = int(status_counts.get(0, 0))
    bad_count = int(status_counts.get(1, 0))
    st.markdown(
        f"""
        <div class="mini-grid">
            <div class="mini-card" style="--accent: var(--green);">
                <strong>Good Score Rows</strong>
                <span>{good_count:,} applicants labelled as Good in training data.</span>
            </div>
            <div class="mini-card" style="--accent: var(--rose);">
                <strong>Bad Score Rows</strong>
                <span>{bad_count:,} applicants labelled as Bad in training data.</span>
            </div>
            <div class="mini-card" style="--accent: var(--blue);">
                <strong>Training Source</strong>
                <span>{training_source}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(dataset.head(200), width="stretch")
