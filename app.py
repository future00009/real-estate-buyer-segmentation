import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Machine Learning Based Real Estate Market Intelligence")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    clients = pd.read_csv("clients.csv")
    properties = pd.read_csv("properties.csv")
    return clients, properties

clients, properties = load_data()

# -----------------------------
# Data cleaning
# -----------------------------
def parse_mixed_date(value):
    if pd.isna(value):
        return pd.NaT
    value = str(value)
    if "/" in value:
        return pd.to_datetime(value, format="%m/%d/%Y", errors="coerce")
    if "-" in value:
        return pd.to_datetime(value, format="%d-%m-%Y", errors="coerce")
    return pd.to_datetime(value, errors="coerce")

clients["dob_parsed"] = clients["date_of_birth"].map(parse_mixed_date)
properties["transaction_date_dt"] = pd.to_datetime(
    properties["transaction_date"], dayfirst=True, errors="coerce"
)
properties["sale_price_num"] = (
    properties["sale_price"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .astype(float)
)

analysis_date = properties["transaction_date_dt"].max()

clients["age"] = (
    (analysis_date - clients["dob_parsed"]).dt.days / 365.25
).round(2)

# -----------------------------
# Build buyer-level features
# -----------------------------
sold = properties[properties["listing_status"].eq("Sold")].copy()
sold["price"] = sold["sale_price_num"]

purchase = sold.groupby("client_ref").agg(
    property_count=("listing_id", "count"),
    total_purchase_value=("price", "sum"),
    avg_purchase_price=("price", "mean"),
    max_purchase_price=("price", "max"),
    total_area_sqft=("floor_area_sqft", "sum"),
    avg_area_sqft=("floor_area_sqft", "mean"),
    towers=("tower_number", "nunique"),
    categories=("unit_category", "nunique")
).reset_index()

purchase = purchase.rename(columns={"client_ref": "client_id"})

# Property category shares
shares = sold.pivot_table(
    index="client_ref",
    columns="unit_category",
    values="listing_id",
    aggfunc="count",
    fill_value=0
)

shares["total_units"] = shares.sum(axis=1)

for column in [c for c in shares.columns if c != "total_units"]:
    shares[column + "_share"] = shares[column] / shares["total_units"]

shares = shares.drop(columns="total_units").reset_index()
shares = shares.rename(columns={"client_ref": "client_id"})

buyer_df = (
    clients
    .merge(purchase, on="client_id", how="left")
    .merge(shares, on="client_id", how="left")
)

numeric_fill = [
    "property_count",
    "total_purchase_value",
    "avg_purchase_price",
    "max_purchase_price",
    "total_area_sqft",
    "avg_area_sqft",
    "towers",
    "categories"
]

for column in numeric_fill:
    buyer_df[column] = buyer_df[column].fillna(0)

for column in shares.columns:
    if column != "client_id":
        buyer_df[column] = buyer_df[column].fillna(0)

buyer_df["price_per_sqft"] = (
    buyer_df["total_purchase_value"]
    / buyer_df["total_area_sqft"].replace(0, np.nan)
).fillna(0)

# -----------------------------
# Machine learning
# -----------------------------
numeric_features = [
    "age",
    "satisfaction_score",
    "property_count",
    "total_purchase_value",
    "avg_purchase_price",
    "max_purchase_price",
    "total_area_sqft",
    "avg_area_sqft",
    "towers",
    "categories",
    "price_per_sqft"
]

numeric_features += [
    c for c in buyer_df.columns if c.endswith("_share")
]

categorical_features = [
    "client_type",
    "gender",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel"
]

X = buyer_df[numeric_features + categorical_features]

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        categorical_features
    )
])

X_scaled = preprocessor.fit_transform(X)

# Evaluate K values
metrics = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=30
    )

    labels = model.fit_predict(X_scaled)

    metrics.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_scaled, labels)
    })

metrics_df = pd.DataFrame(metrics)

# Four-cluster model, matching the PRD's four-segment structure.
FINAL_K = 4

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=30
)

buyer_df["cluster"] = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(
    X_scaled,
    buyer_df["cluster"]
)

# Hierarchical validation
hierarchical = AgglomerativeClustering(
    n_clusters=FINAL_K,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

# -----------------------------
# Segment naming
# -----------------------------
cluster_profile = buyer_df.groupby("cluster").agg(
    buyers=("client_id", "count"),
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_properties=("property_count", "mean"),
    avg_total_value=("total_purchase_value", "mean"),
    investment_share=(
        "acquisition_purpose",
        lambda x: (x == "Investment").mean()
    ),
    loan_share=(
        "loan_applied",
        lambda x: (x == "Yes").mean()
    ),
    company_share=(
        "client_type",
        lambda x: (x == "Company").mean()
    )
).reset_index()

# Descriptive names derived from observed purchase behavior.
ranking = cluster_profile.sort_values(
    ["avg_properties", "avg_total_value"],
    ascending=[False, False]
).reset_index(drop=True)

segment_names = {}

for i, row in ranking.iterrows():
    cluster = int(row["cluster"])

    if i == 0:
        segment_names[cluster] = "Repeat / High-Value Buyers"
    elif row["avg_total_value"] < cluster_profile["avg_total_value"].median():
        segment_names[cluster] = "Lower-Value Residential Buyers"
    elif row["company_share"] > 0.20:
        segment_names[cluster] = "Mixed Residential-Commercial Buyers"
    else:
        segment_names[cluster] = "Standard Residential Investors"

buyer_df["segment"] = buyer_df["cluster"].map(segment_names)

cluster_profile["segment"] = cluster_profile["cluster"].map(segment_names)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

filtered = buyer_df.copy()

for column, label in [
    ("country", "Country"),
    ("region", "Region"),
    ("acquisition_purpose", "Acquisition Purpose"),
    ("client_type", "Client Type")
]:
    values = sorted(filtered[column].dropna().unique())

    selected = st.sidebar.multiselect(
        label,
        values,
        default=values
    )

    filtered = filtered[filtered[column].isin(selected)]

# -----------------------------
# KPI section
# -----------------------------
st.subheader("Buyer Segmentation Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Buyers",
    f"{len(filtered):,}"
)

c2.metric(
    "Total Purchase Value",
    f"${filtered['total_purchase_value'].sum():,.0f}"
)

c3.metric(
    "Average Satisfaction",
    f"{filtered['satisfaction_score'].mean():.2f}"
)

investment_percentage = (
    filtered["acquisition_purpose"].eq("Investment").mean() * 100
)

c4.metric(
    "Investment Buyers",
    f"{investment_percentage:.1f}%"
)

# -----------------------------
# Segment distribution
# -----------------------------
st.subheader("Buyer Segments")

segment_counts = (
    filtered["segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["segment", "buyers"]

fig = px.bar(
    segment_counts,
    x="segment",
    y="buyers",
    title="Buyer Segment Distribution"
)

fig.update_xaxes(tickangle=-25)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Investor behavior
# -----------------------------
st.subheader("Investor Behavior")

fig = px.scatter(
    filtered,
    x="property_count",
    y="total_purchase_value",
    size="total_area_sqft",
    color="segment",
    hover_data=[
        "client_id",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied"
    ],
    title="Purchase Count vs Total Purchase Value"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Acquisition purpose
# -----------------------------
purpose = pd.crosstab(
    filtered["segment"],
    filtered["acquisition_purpose"],
    normalize="index"
).reset_index()

purpose_long = purpose.melt(
    id_vars="segment",
    var_name="purpose",
    value_name="share"
)

fig = px.bar(
    purpose_long,
    x="segment",
    y="share",
    color="purpose",
    barmode="stack",
    title="Acquisition Purpose by Segment"
)

fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(tickangle=-25)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Geographic analysis
# -----------------------------
st.subheader("Geographic Buyer Analysis")

geo = (
    filtered
    .groupby(["country", "region"])
    .agg(
        buyers=("client_id", "count"),
        purchase_value=("total_purchase_value", "sum")
    )
    .reset_index()
    .sort_values("buyers", ascending=False)
)

st.dataframe(
    geo,
    use_container_width=True
)

# -----------------------------
# Segment insights
# -----------------------------
st.subheader("Segment Insights")

display_profile = cluster_profile.copy()

for column in [
    "avg_age",
    "avg_satisfaction",
    "avg_properties",
    "avg_total_value",
    "investment_share",
    "loan_share",
    "company_share"
]:
    if column in display_profile.columns:
        display_profile[column] = display_profile[column].round(3)

st.dataframe(
    display_profile,
    use_container_width=True
)

# -----------------------------
# Model validation
# -----------------------------
st.subheader("Clustering Validation")

col1, col2 = st.columns(2)

with col1:
    fig = px.line(
        metrics_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.line(
        metrics_df,
        x="K",
        y="Silhouette",
        markers=True,
        title="Silhouette Score"
    )
    fig.add_vline(
        x=FINAL_K,
        line_dash="dash"
    )
    st.plotly_chart(fig, use_container_width=True)

m1, m2 = st.columns(2)

m1.metric(
    "K-Means K=4 Silhouette",
    f"{kmeans_silhouette:.4f}"
)

m2.metric(
    "Hierarchical K=4 Silhouette",
    f"{hierarchical_silhouette:.4f}"
)

st.info(
    "The four-cluster structure follows the project PRD. "
    "The silhouette score is moderate, so these should be interpreted "
    "as practical analytical buyer segments rather than perfectly "
    "separated natural classes."
)

# -----------------------------
# Raw/clustered data
# -----------------------------
st.subheader("Buyer-Level Clustered Data")

st.dataframe(
    filtered,
    use_container_width=True
)

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered Buyer Data",
    csv,
    "buyer_segmentation_results.csv",
    "text/csv"
)

st.caption(
    "Project: Machine Learning Based Buyer Segmentation and Investment Profiling "
    "for Real Estate Market Intelligence"
)
