from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier

st.title("Credit Card Customer Analysis")

st.markdown("""
This dashboard explores customer credit behavior, 
segments users using clustering, and predicts revolving credit usage.
""")

#Cache expensive computations or potential ones
LOAD_URL = ('https://github.com/gentralg-Gitcoding/fullstack-projects/blob/main/capstoneProject/CC_GENERAL.csv')
@st.cache_data
def load_og_data():
    return pd.read_csv(LOAD_URL)

og_data = load_og_data()

#Impute data for clean data analysis
@st.cache_resource
def preprocess_for_clean_data(df, features):
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(df[features])
    return imputed_data

knn_neighbors = [
        'PAYMENTS', 
        'BALANCE', 
        'ONEOFF_PURCHASES', 
        'INSTALLMENTS_PURCHASES', 
        'CREDIT_LIMIT', 
        'TENURE', 
        'MINIMUM_PAYMENTS'
    ]
og_data[knn_neighbors] = preprocess_for_clean_data(og_data, knn_neighbors)

#Show summary overview of what the data is(Loading the CSV)
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", og_data.shape[0])
col2.metric("Features", og_data.shape[1])

#Show optional raw data(Sample Overview)
with st.expander("View raw data(top 100 rows)"):
    st.dataframe(og_data.head(100))

#Show how each feature looks in plots(Visualize the Data)
st.subheader("Feature Distributions")

numeric_features = [
    'BALANCE', 
    'PURCHASES', 
    'ONEOFF_PURCHASES', 
    'INSTALLMENTS_PURCHASES', 
    'CASH_ADVANCE', 
    'CREDIT_LIMIT', 
    'PAYMENTS', 
    'MINIMUM_PAYMENTS',
    'CASH_ADVANCE_TRX', 
    'PURCHASES_TRX', 
    'TENURE', 
    'BALANCE_FREQUENCY', 
    'PURCHASES_FREQUENCY', 
    'ONEOFF_PURCHASES_FREQUENCY', 
    'PURCHASES_INSTALLMENTS_FREQUENCY', 
    'CASH_ADVANCE_FREQUENCY'
]

view_feature = st.selectbox(
    "Select a numeric feature to explore",
    numeric_features
)

#Let User check if they want the transformed view of the data used for the models
#Visualizes less skewed data
use_power = st.checkbox("Apply Power Transformation", value=False)

#Cache our PowerTransformer to save runtime. I noticed processing slowing down very quickly
@st.cache_resource
def get_power_transformer(data):
    pt = PowerTransformer()
    pt.fit(data)
    return pt

#We have to reshape a single series feature due to PowerTransform wanting it in a (XXXX, 1) shape not a (1,XXXX) to show it as 1 feature and multiple rows
plot_data = og_data[view_feature].values.reshape(-1,1) if use_power else og_data[view_feature]
if use_power:
    power_transformer = get_power_transformer(plot_data)
    plot_data = pd.DataFrame(
        power_transformer.transform(plot_data)
            .ravel()
        )

# Plot Histogram
fig, ax = plt.subplots()
ax.hist(plot_data, bins=50, edgecolor='black')
ax.set_title(f"Hist: Distribution of {view_feature}" + ("(Power Transformed)" if use_power else ""))
ax.set_xlabel(view_feature)
ax.set_ylabel("Count")

st.pyplot(fig)

#Noting down the IQR outliers for our graphs 
q1 = og_data[view_feature].quantile(0.25)
q3 = og_data[view_feature].quantile(0.75)
iqr = q3 - q1

outliers = og_data[(og_data[view_feature] < q1 - 1.5 * iqr) | 
              (og_data[view_feature] > q3 + 1.5 * iqr)]

st.caption(f"Detected IQR(1.5) outliers: {outliers.shape[0]} records")

# Summary statistics on the selected feature
with st.expander("Summary statistics"):
    st.write(plot_data.describe())


# Do some dimensionality reduction for visualization
power_data = pd.DataFrame(
    get_power_transformer(
        og_data[numeric_features]
    ).transform(og_data[numeric_features]),
    columns=numeric_features,
    index=og_data[numeric_features].index
)

# PCA for visualization only
@st.cache_resource
def get_pca(data):
    pca = PCA(n_components=2, random_state=320)
    pca.fit(data)
    return pca

X_pca_2d = get_pca(power_data).transform(power_data)

# KMeans on scaled full feature set
@st.cache_resource
def get_kmeans(data, k=2):
    kmeans = KMeans(n_clusters=k, random_state=320)
    kmeans.fit(data)
    return kmeans

kmeans = get_kmeans(power_data)
labels = kmeans.labels_

#Plot our PCA reduction
fig, ax = plt.subplots()
ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("PCA 2D Projection with KMeans Clusters(k=2)")
st.pyplot(plt)


#Create Kmeans profiles for slider interaction
st.subheader("KMeans Cluster Profiles")

k = st.slider(
    "Select number of clusters (k)",
    min_value=2,
    max_value=8,
    value=2,
    step=1
)

slider_kmeans = get_kmeans(power_data, k)
slider_labels = slider_kmeans.labels_

clustered_df = og_data.loc[power_data.index].copy()
clustered_df["cluster"] = slider_labels

#Display a Dataframe of averages for cluster groups
profile = (
    clustered_df
    .groupby("cluster")[numeric_features]
    .mean()
    .round(2)
)

st.dataframe(profile)
st.caption('Dataframe: Averages per cluster')

st.markdown('BarChart: Size per Cluster')
cluster_sizes = clustered_df["cluster"].value_counts().sort_index()
st.bar_chart(cluster_sizes)


#Create a dataframe for the model pipelines and our target feature
model_df = og_data.copy(deep=True)

ca_threshold = model_df['CASH_ADVANCE'].quantile(0.5)
prc_threshold = model_df['PRC_FULL_PAYMENT'].quantile(0.5)

model_df['REVOLVER'] = (
    (model_df['CASH_ADVANCE'] >= ca_threshold) &
    (model_df['PRC_FULL_PAYMENT'] <= prc_threshold)
).astype(bool)

y = model_df['REVOLVER']
X = model_df.drop(columns=['REVOLVER','CUST_ID','CASH_ADVANCE', 'PRC_FULL_PAYMENT'])

#Model Predictions Section and threshold tuning 
@st.cache_resource
def get_gradient_boosting_classifier(X, y):
    gb_pipeline = Pipeline([
        ('knn_imputer', KNNImputer()),
        ('gradient_classifier', GradientBoostingClassifier(random_state=320))
    ])
    gb_pipeline.fit(X, y)
    return gb_pipeline

st.subheader("Revolver Risk Prediction")
st.markdown("""
    This section estimates the probability that a customer is a **revolver**
    (based on spending and payment behavior).
    """)

threshold = st.slider(
    "Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.63,     #Best threshold from GradientBoostingClassifier
    step=0.01,
    help="Higher threshold = fewer but more confident revolver predictions"
)

probs = get_gradient_boosting_classifier(X,y).predict_proba(X)[:, 1]    #Class 1 meaning they are revolvers
revolver_preds = (probs >= threshold)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Revolver Risk",
    f"{probs.mean():.2%}"
)

col2.metric(
    "Number of High-Risk Customers",
    f"{revolver_preds.sum()}"
)

col3.metric(
    "High-Risk Rate",
    f"{revolver_preds.mean():.2%}"
)

if st.checkbox("Show High-Risk Customers"):
    risk_df = og_data.copy()
    risk_df["revolver_probability"] = probs
    risk_df["revolver_pred"] = revolver_preds

    st.dataframe(
        risk_df[risk_df["revolver_pred"]]
        .sort_values("revolver_probability", ascending=False)
    )

# Show clusters and how risky they are of being revolvers
st.header("Cluster x Revolver Risk")

#remake so user doesn't have to scroll up 
k = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=8,
    value=2,
    step=1
)

slider_kmeans = get_kmeans(power_data, k)
clusters = slider_kmeans.predict(power_data)

#Copy og_data and add back in clusters, predictions and labels 
cluster_risk_df = og_data.copy()
cluster_risk_df["cluster"] = clusters
cluster_risk_df["revolver_probability"] = probs
cluster_risk_df["revolver_pred"] = revolver_preds

#Create df aggregation for customers, probability, and revolver prediction
cluster_summary = (
    cluster_risk_df
    .groupby("cluster")
    .agg(
        customers=("cluster", "count"),
        avg_risk=("revolver_probability", "mean"),
        high_risk_rate=("revolver_pred", "mean")
    )
    .reset_index()
)

#Show df and scatter plot
st.subheader("Cluster Risk Profile")
st.dataframe(cluster_summary)

fig, ax = plt.subplots()
ax.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=clusters
)

ax.set_title("Customer Clusters (PCA Projection)")
st.pyplot(fig)


st.header("Final Conclusion")

st.markdown("""
This analysis displays **behavioral patterns** and **risky chances**
to identify customers most likely to revolve balances and accrue interest.
The goal is to support **targeted revolving users** and **risk-awareness**.
""")

st.subheader("Key Findings")
highest_risk_cluster = cluster_summary[cluster_summary['high_risk_rate'].max() == cluster_summary['high_risk_rate']]['cluster'][0]
# st.write(highest_risk_cluster)
percentage_of_customers = ((cluster_summary[highest_risk_cluster == cluster_summary['cluster']]['customers'])/cluster_summary['customers'].sum())[0]
# st.write(percentage_of_customers)

st.markdown(f"""
- **{revolver_preds.mean():.1%} of customers** are classified as high revolver risk
  at the selected decision threshold.
- **Cluster {highest_risk_cluster}** contains a disproportionate share of high-risk customers,
  despite representing only **{percentage_of_customers:.1%}** of the population.
- Revolver risk is most strongly associated with:
  - High **cash advance usage**
  - Low **full payment ratio**
  - Higher **credit utilization**
""")

st.subheader("Cluster-Level Risk Interpretation")

st.markdown("""
Overall revolver risk on behavioral patterns shows us meaningful differences:

- **Cluster 0 (Revolvers)**  
  High cash advance usage and low full payment rates.
  This cluster accounts for the majority of high-risk predictions.

- **Cluster 1 (Transactors)**  
  Low revolver risk, consistent full payments, minimal cash advances.
  These customers are strong candidates for **rewards and retention** strategies.
""")

st.subheader("Recommendations")

st.markdown("""
**1. Proactive Risk Management**
- Monitor customers in high-risk clusters with higher revolver usage
- Be careful considering credit line increases for these groups

**2. Targeted Interventions**
- Offer payment plans or balance alerts to high-risk customers
- Reduce reliance on blanket credit policies

**3. Growth Opportunities**
- Preserve and reward low-risk clusters to prevent future revolver behavior
- Use cluster assignment as an input into lifecycle management
""")

