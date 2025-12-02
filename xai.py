import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import FeatureAgglomeration
import xgboost as xgb
from scipy.stats import spearmanr
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data.csv')
print(f"Original dataset shape: {df.shape}")

# Delete all instances when 'spreads' values are empty
df = df.dropna(subset=['spreads'])

# Calculate how many instances to keep (1/10 of the original)
total_rows = len(df)
rows_to_keep = total_rows // 10

# Count missing values per row
missing_counts = df.isnull().sum(axis=1)

# Sort by number of missing values (descending) and keep only rows_to_keep
df = df.iloc[missing_counts.argsort()[:rows_to_keep]]

# Process all columns to ensure they are numeric
for col in df.columns:
    if col == 'spreads':
        continue  # Skip the target column
    
    # Check if column contains string data
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        # Special handling for date column
        if col == 'last_loan_issuance':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').astype(np.int64) // 10**9
            except:
                # If date conversion fails, use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna('missing').astype(str))
        else:
            # Label encoding for all other string columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('missing').astype(str))
    
    # Ensure the column is numeric
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill empty cells with zeros
df = df.fillna(0)

print(f"Reduced dataset shape: {df.shape}")

# Prepare X and y
X = df.drop('spreads', axis=1)
y = df['spreads']

# Verify all columns are numeric
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Column {col} is not numeric: {X[col].dtype}")
        # Convert any remaining non-numeric columns
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# Shuffle data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X.iloc[indices].reset_index(drop=True)
y = y.iloc[indices].reset_index(drop=True)

# Initialize feature selection methods and results storage
methods = ['Random Forest', 'XGBoost', 'Feature Agglomeration', 'High Variance', 'Spearman Correlation', 'LIME-RF', 'LIME-XGBoost']
results = {method: {'cv10': 0, 'cv9': 0, 'top5_cv10': [], 'top4_cv9': []} for method in methods}

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
feature_importances = rf.feature_importances_
top_10_indices = np.argsort(feature_importances)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest feature and remove it from the dataset
top_feature_idx = np.argsort(feature_importances)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Retrain on reduced dataset
rf_reduced = RandomForestRegressor(random_state=42)
rf_reduced.fit(X_reduced, y)
feature_importances_reduced = rf_reduced.feature_importances_
top_9_indices_reduced = np.argsort(feature_importances_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices_reduced]

results['Random Forest']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['Random Forest']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['Random Forest']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['Random Forest']['top4_cv9'] = X_reduced.columns[top_9_indices_reduced[:4]].tolist()

# 2. XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)
feature_importances = xgb_model.feature_importances_
top_10_indices = np.argsort(feature_importances)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest feature and remove it from the dataset
top_feature_idx = np.argsort(feature_importances)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Retrain on reduced dataset
xgb_model_reduced = xgb.XGBRegressor(random_state=42)
xgb_model_reduced.fit(X_reduced, y)
feature_importances_reduced = xgb_model_reduced.feature_importances_
top_9_indices_reduced = np.argsort(feature_importances_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices_reduced]

results['XGBoost']['cv10'] = np.mean(cross_val_score(xgb_model, X_cv10, y, cv=kf, scoring='r2'))
results['XGBoost']['cv9'] = np.mean(cross_val_score(xgb_model_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['XGBoost']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['XGBoost']['top4_cv9'] = X_reduced.columns[top_9_indices_reduced[:4]].tolist()

# 3. Feature Agglomeration
fa = FeatureAgglomeration(n_clusters=X.shape[1] // 2)
fa.fit(X)

# Get cluster assignments for each feature
clusters = fa.labels_

# Calculate importance of each cluster using variance
cluster_importances = []
for i in range(max(clusters) + 1):
    cluster_features = X.iloc[:, clusters == i].values
    cluster_importances.append(np.sum(np.var(cluster_features, axis=0)))

# Sort clusters by importance
sorted_clusters = np.argsort(cluster_importances)[::-1]

# Get features from each cluster by importance
top_features = []
for cluster_idx in sorted_clusters:
    features_in_cluster = np.where(clusters == cluster_idx)[0]
    top_features.extend(features_in_cluster)

top_10_indices = top_features[:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest feature and remove it from the dataset
top_feature_idx = top_features[0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Rerun Feature Agglomeration on reduced dataset
fa_reduced = FeatureAgglomeration(n_clusters=X_reduced.shape[1] // 2)
fa_reduced.fit(X_reduced)
clusters_reduced = fa_reduced.labels_

# Calculate importance of each cluster for reduced dataset
cluster_importances_reduced = []
for i in range(max(clusters_reduced) + 1):
    cluster_features = X_reduced.iloc[:, clusters_reduced == i].values    
    cluster_importances_reduced.append(np.sum(np.var(cluster_features, axis=0)))

# Sort clusters by importance for reduced dataset
sorted_clusters_reduced = np.argsort(cluster_importances_reduced)[::-1]

# Get features from each cluster by importance for reduced dataset
top_features_reduced = []
for cluster_idx in sorted_clusters_reduced:
    features_in_cluster = np.where(clusters_reduced == cluster_idx)[0]    
    top_features_reduced.extend(features_in_cluster)

top_9_indices_reduced = top_features_reduced[:9]
X_cv9 = X_reduced.iloc[:, top_9_indices_reduced]

rf = RandomForestRegressor(random_state=42)
rf_reduced = RandomForestRegressor(random_state=42)
results['Feature Agglomeration']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['Feature Agglomeration']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['Feature Agglomeration']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['Feature Agglomeration']['top4_cv9'] = X_reduced.columns[top_9_indices_reduced[:4]].tolist()

# 4. High Variance Gene Selection (HVGS)
variances = X.var().values
top_10_indices = np.argsort(variances)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest variance feature and remove it from the dataset
top_feature_idx = np.argsort(variances)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Recalculate variances on reduced dataset
variances_reduced = X_reduced.var().values
top_9_indices_reduced = np.argsort(variances_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices_reduced]

rf = RandomForestRegressor(random_state=42)
rf_reduced = RandomForestRegressor(random_state=42)
results['High Variance']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['High Variance']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['High Variance']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['High Variance']['top4_cv9'] = X_reduced.columns[top_9_indices_reduced[:4]].tolist()

# 5. Spearman Correlation
corr_values = []
for col in X.columns:
    corr, _ = spearmanr(X[col], y)
    corr_values.append(abs(corr) if not np.isnan(corr) else 0)  # Use absolute correlation

corr_values = np.array(corr_values)
top_10_indices = np.argsort(corr_values)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest correlation feature and remove it from the dataset
top_feature_idx = np.argsort(corr_values)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Recalculate correlations on reduced dataset
corr_values_reduced = []
for col in X_reduced.columns:
    corr, _ = spearmanr(X_reduced[col], y)
    corr_values_reduced.append(abs(corr) if not np.isnan(corr) else 0)
corr_values_reduced = np.array(corr_values_reduced)
top_9_indices_reduced = np.argsort(corr_values_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices_reduced]

rf = RandomForestRegressor(random_state=42)
rf_reduced = RandomForestRegressor(random_state=42)
results['Spearman Correlation']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['Spearman Correlation']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['Spearman Correlation']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['Spearman Correlation']['top4_cv9'] = X_reduced.columns[top_9_indices_reduced[:4]].tolist()

# 6. LIME-RF
# Using cross-validation approach for LIME
sample_size = min(50, len(X))  # Use up to 50 samples for LIME calculations

# Train RF model for LIME
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Create LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    X.values, 
    feature_names=X.columns.tolist(),
    mode='regression',
    random_state=42
)

# Generate LIME explanations and get feature importances
lime_rf_importance = np.zeros(X.shape[1])
for i in range(sample_size):
    exp = explainer.explain_instance(
        X.iloc[i].values, 
        rf_model.predict, 
        num_features=10
    )
    # Process feature importance from LIME
    for feature_idx, importance in exp.local_exp[1]:
        lime_rf_importance[feature_idx] += abs(importance)

# Normalize importance values
lime_rf_importance = lime_rf_importance / sample_size

# Get top features based on LIME-RF
top_10_indices = np.argsort(lime_rf_importance)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest feature and remove it from the dataset
top_feature_idx = np.argsort(lime_rf_importance)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Create reduced model and explainer
rf_model_reduced = RandomForestRegressor(random_state=42)
rf_model_reduced.fit(X_reduced, y)

explainer_reduced = lime_tabular.LimeTabularExplainer(
    X_reduced.values,
    feature_names=X_reduced.columns.tolist(),
    mode='regression',
    random_state=42
)

# Calculate importances for reduced dataset
lime_rf_importance_reduced = np.zeros(X_reduced.shape[1])
for i in range(sample_size):
    try:
        exp = explainer_reduced.explain_instance(
            X_reduced.iloc[i].values,
            rf_model_reduced.predict,
            num_features=9
        )
        for feature_idx, importance in exp.local_exp[1]:
            lime_rf_importance_reduced[feature_idx] += abs(importance)
    except:
        continue

# Normalize importance values
lime_rf_importance_reduced = lime_rf_importance_reduced / sample_size

# Get top features for CV9
top_9_indices = np.argsort(lime_rf_importance_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices]

rf = RandomForestRegressor(random_state=42)
rf_reduced = RandomForestRegressor(random_state=42)
results['LIME-RF']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['LIME-RF']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['LIME-RF']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['LIME-RF']['top4_cv9'] = X_reduced.columns[top_9_indices[:4]].tolist()

# 7. LIME-XGBoost
# Train XGBoost model for LIME
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)

# Generate LIME explanations for XGBoost
lime_xgb_importance = np.zeros(X.shape[1])
for i in range(sample_size):
    exp = explainer.explain_instance(
        X.iloc[i].values, 
        xgb_model.predict, 
        num_features=10
    )
    for feature_idx, importance in exp.local_exp[1]:
        lime_xgb_importance[feature_idx] += abs(importance)

# Normalize importance values
lime_xgb_importance = lime_xgb_importance / sample_size

# Get top features based on LIME-XGBoost
top_10_indices = np.argsort(lime_xgb_importance)[::-1][:10]
X_cv10 = X.iloc[:, top_10_indices]

# Get the highest feature and remove it from the dataset
top_feature_idx = np.argsort(lime_xgb_importance)[::-1][0]
X_reduced = X.drop(X.columns[top_feature_idx], axis=1)

# Create reduced model
xgb_model_reduced = xgb.XGBRegressor(random_state=42)
xgb_model_reduced.fit(X_reduced, y)

# Calculate importances for reduced dataset
lime_xgb_importance_reduced = np.zeros(X_reduced.shape[1])
for i in range(sample_size):
    try:
        exp = explainer_reduced.explain_instance(
            X_reduced.iloc[i].values,
            xgb_model_reduced.predict,
            num_features=9
        )
        for feature_idx, importance in exp.local_exp[1]:
            lime_xgb_importance_reduced[feature_idx] += abs(importance)
    except:
        continue

# Normalize importance values
lime_xgb_importance_reduced = lime_xgb_importance_reduced / sample_size

# Get top features for CV9
top_9_indices = np.argsort(lime_xgb_importance_reduced)[::-1][:9]
X_cv9 = X_reduced.iloc[:, top_9_indices]

rf = RandomForestRegressor(random_state=42)
rf_reduced = RandomForestRegressor(random_state=42)
results['LIME-XGBoost']['cv10'] = np.mean(cross_val_score(rf, X_cv10, y, cv=kf, scoring='r2'))
results['LIME-XGBoost']['cv9'] = np.mean(cross_val_score(rf_reduced, X_cv9, y, cv=kf, scoring='r2'))
results['LIME-XGBoost']['top5_cv10'] = X.columns[top_10_indices[:5]].tolist()
results['LIME-XGBoost']['top4_cv9'] = X_reduced.columns[top_9_indices[:4]].tolist()

# Create summary table
summary = []
for method in methods:
    summary.append({
        'Method': method,
        'CV10 R-squared': f"{results[method]['cv10']:.4f}",  # Format to 4 decimal places
        'CV9 R-squared': f"{results[method]['cv9']:.4f}",    # Format to 4 decimal places
        'Top 5 Features (CV10)': ', '.join(results[method]['top5_cv10']),
        'Top 4 Features (CV9)': ', '.join(results[method]['top4_cv9'])
    })

summary_df = pd.DataFrame(summary)
print("\nSummary Results:")
print(summary_df)

# Save the summary table as CSV
summary_df.to_csv('result.csv', index=False)
print("Summary results saved to 'result.csv'")

# Find the best performing model based on CV10 R-squared
best_method = max(results, key=lambda x: results[x]['cv10'])
print(f"\nBest performing method: {best_method}")
print(f"CV10 R-squared: {results[best_method]['cv10']:.4f}")
print(f"Top 5 features: {', '.join(results[best_method]['top5_cv10'])}")
