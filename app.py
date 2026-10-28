import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation Using Machine Learning")
st.caption("An AI/ML mini project using K-Means clustering")

DATA_URL = "https://raw.githubusercontent.com/MahimaSriK/Mall-Customer-Segmentation-Dataset/main/Mall_Customers.csv"

@st.cache_data
def load_online_data():
    return pd.read_csv(DATA_URL)

with st.sidebar:
    st.header("Project Controls")
    st.write("Choose the number of customer groups.")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=5)
    st.markdown("---")
    st.write("**Algorithm:** K-Means")
    st.write("**Features:** Annual income and spending score")

try:
    df = load_online_data()
    st.sidebar.success("Sample dataset loaded")
except Exception:
    st.warning("Could not download the sample dataset. Upload Mall_Customers.csv below.")
    uploaded = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])
    if uploaded is None:
        st.info("Download the CSV from the dataset link in README.md, then upload it here.")
        st.stop()
    df = pd.read_csv(uploaded)

# Standardize common dataset column names
df.columns = [str(c).strip() for c in df.columns]
rename_map = {}
for col in df.columns:
    low = col.lower().strip()
    if low in ["genre", "gender"]:
        rename_map[col] = "Gender"
    elif low == "age":
        rename_map[col] = "Age"
    elif "annual income" in low or "annual_income" in low:
        rename_map[col] = "Annual Income"
    elif "spending score" in low or "spending_score" in low:
        rename_map[col] = "Spending Score"
    elif low in ["customerid", "customer id", "customer_id"]:
        rename_map[col] = "CustomerID"
df = df.rename(columns=rename_map)

required = ["Annual Income", "Spending Score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}. Please use the Mall Customers dataset.")
    st.stop()

df["Annual Income"] = pd.to_numeric(df["Annual Income"], errors="coerce")
df["Spending Score"] = pd.to_numeric(df["Spending Score"], errors="coerce")
df = df.dropna(subset=required).copy()
if len(df) < k:
    st.error("There are fewer valid rows than clusters. Upload a larger dataset or reduce K.")
    st.stop()

features = df[["Annual Income", "Spending Score"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled)
score = silhouette_score(scaled, df["Cluster"]) if 1 < k < len(df) else None

c1, c2, c3 = st.columns(3)
c1.metric("Customers analysed", f"{len(df):,}")
c2.metric("Customer groups", str(k))
c3.metric("Silhouette score", f"{score:.3f}" if score is not None else "N/A")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Explore Customers", "🧠 Predict a Group"])

with tab1:
    st.subheader("Customer Groups")
    st.write("Each point represents a customer. Groups are formed using annual income and spending score.")
    fig, ax = plt.subplots(figsize=(9, 5))
    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        ax.scatter(subset["Annual Income"], subset["Spending Score"], label=f"Cluster {cluster_id + 1}", alpha=0.8)
    ax.set_xlabel("Annual Income (thousands of dollars)")
    ax.set_ylabel("Spending Score (1–100)")
    ax.set_title("Customer Segmentation using K-Means")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Group Summary")
    summary = df.groupby("Cluster").agg(
        Customers=("Spending Score", "size"),
        Average_Income=("Annual Income", "mean"),
        Average_Spending_Score=("Spending Score", "mean")
    ).round(2).reset_index()
    summary["Cluster"] = summary["Cluster"] + 1
    summary = summary.rename(columns={
        "Average_Income": "Average Annual Income (k$)",
        "Average_Spending_Score": "Average Spending Score"
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Model Evaluation")
    if score is not None:
        st.write(f"**Silhouette score:** {score:.3f}. This measures how well-separated the clusters are; values closer to 1 generally indicate better separation.")
    st.write("K-Means is an unsupervised learning algorithm, so it groups similar customers rather than predicting a pre-labelled answer.")

with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.subheader("Filter Customers")
    min_income = int(df["Annual Income"].min())
    max_income = int(df["Annual Income"].max())
    income_range = st.slider("Annual income range (k$)", min_value=min_income, max_value=max_income,
                             value=(min_income, max_income))
    filtered = df[df["Annual Income"].between(income_range[0], income_range[1])]
    st.write(f"Showing **{len(filtered)}** customers")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download clustered results as CSV",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="customer_segments.csv", mime="text/csv")

with tab3:
    st.subheader("Find the likely customer group")
    st.write("Enter example customer details to assign them to the nearest learned cluster.")
    income = st.number_input("Annual income (k$)", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
    spending = st.slider("Spending score", min_value=1, max_value=100, value=50)
    if st.button("Identify Customer Group", type="primary"):
        new_scaled = scaler.transform([[income, spending]])
        predicted_cluster = int(model.predict(new_scaled)[0])
        group_data = df[df["Cluster"] == predicted_cluster]
        avg_income = group_data["Annual Income"].mean()
        avg_spending = group_data["Spending Score"].mean()
        st.success(f"This customer belongs to **Cluster {predicted_cluster + 1}**.")
        st.write(f"This group contains **{len(group_data)}** customers in the dataset.")
        st.write(f"Group average income: **{avg_income:.1f} k$**")
        st.write(f"Group average spending score: **{avg_spending:.1f}/100**")

st.markdown("---")
st.caption("Mini Project | Python • Pandas • Scikit-learn • Streamlit • Matplotlib")
