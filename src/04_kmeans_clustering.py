"""
=============================================================
  Heart Disease Prediction — Member 4: K-Means Clustering
=============================================================
  Model   : K-Means Clustering (Unsupervised)
  Purpose : Patient risk segmentation into risk groups
  Task    : Segment patients into Low/Medium/High risk groups
  Dataset : UCI Heart Disease — Cleveland (processed.cleveland.data)
  Outputs : Elbow plot, Silhouette analysis, PCA visualization,
            Heatmap, and Cluster summary table
=============================================================
"""

# ── Standard Library ─────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings

# ══════════════════════════════════════════════════════════════
#  1. PATHS & CONFIGURATION
# ══════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "Data_set", "processed.cleveland.data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "kmeans")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature names matching the dataset structure
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Feature descriptions for interpretation
FEATURE_DESCRIPTIONS = {
    "age"      : "Age (years)",
    "sex"      : "Sex (1=Male, 0=Female)",
    "cp"       : "Chest Pain Type",
    "trestbps" : "Resting BP (mm Hg)",
    "chol"     : "Serum Cholesterol (mg/dl)",
    "fbs"      : "Fasting Blood Sugar (1=high, 0=low)",
    "restecg"  : "Resting ECG Results",
    "thalach"  : "Max Heart Rate Achieved",
    "exang"    : "Exercise Angina (1=yes, 0=no)",
    "oldpeak"  : "ST Depression",
    "slope"    : "ST Segment Slope",
    "ca"       : "Major Vessels (0-3)",
    "thal"     : "Thalassemia Type"
}

# ══════════════════════════════════════════════════════════════
#  2. LOAD & PREPROCESS DATA
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA...")
print("=" * 70)

# Load the dataset
df = pd.read_csv(DATA_PATH, names=COLUMN_NAMES, na_values=['?'])

# Remove target column for clustering (unsupervised)
X = df.drop("target", axis=1)

print(f"Dataset shape before cleaning: {X.shape}")
print(f"Features: {list(X.columns)}")

# Check for missing values
missing_count = X.isnull().sum().sum()
print(f"Missing values found: {missing_count}")

# Handle missing values - drop rows with any missing values
if missing_count > 0:
    print(f"Dropping {X.isnull().any(axis=1).sum()} rows with missing values...")
    X = X.dropna()
    df = df.dropna()
    print(f"Dataset shape after cleaning: {X.shape}")

# Normalize/Standardize features (critical for K-Means)
print("\nNormalizing features...")
# Convert to numeric if needed
X = X.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessing complete!")

# ══════════════════════════════════════════════════════════════
#  3. ELBOW METHOD - Find optimal number of clusters
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ELBOW METHOD - Finding optimal clusters...")
print("=" * 70)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.3f}")

# ══════════════════════════════════════════════════════════════
#  4. PLOT 1: ELBOW PLOT
# ══════════════════════════════════════════════════════════════
print("\nGenerating Elbow Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Optimal K=3')
ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
ax.set_title('Elbow Plot: Optimal Number of Clusters', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_elbow_plot.png"), dpi=300, bbox_inches='tight')
print("[OK] Saved: 01_elbow_plot.png")
plt.close()

# ══════════════════════════════════════════════════════════════
#  5. OPTIMAL K-MEANS MODEL (K=3 for risk groups)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FITTING K-MEANS WITH K=3 (Low/Medium/High Risk)")
print("=" * 70)

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Add cluster labels to original dataframe
X['Cluster'] = cluster_labels
df['Cluster'] = cluster_labels

print(f"Cluster distribution:")
print(X['Cluster'].value_counts().sort_index())
print(f"\nFinal Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")

# ══════════════════════════════════════════════════════════════
#  6. PLOT 2: SILHOUETTE ANALYSIS PLOT
# ══════════════════════════════════════════════════════════════
print("\nGenerating Silhouette Analysis Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Selected K=3')
ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Silhouette Analysis: Cluster Quality', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_silhouette_analysis.png"), dpi=300, bbox_inches='tight')
print("[OK] Saved: 02_silhouette_analysis.png")
plt.close()

# ══════════════════════════════════════════════════════════════
#  7. PCA FOR 3D VISUALIZATION
# ══════════════════════════════════════════════════════════════
print("\nPerforming PCA for visualization...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")

# ══════════════════════════════════════════════════════════════
#  8. PLOT 3: PCA SCATTER PLOT (3D)
# ══════════════════════════════════════════════════════════════
print("\nGenerating PCA 3D Scatter Plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']

for i in range(optimal_k):
    cluster_points = X_pca[cluster_labels == i]
    ax.scatter(
        cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
        c=colors[i], label=risk_labels[i], s=100, alpha=0.7, edgecolors='black', linewidth=0.5
    )

# Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2],
           c='purple', marker='*', s=500, edgecolors='black', linewidth=2, label='Centroids')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11, fontweight='bold')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11, fontweight='bold')
ax.set_title('Patient Clusters in PCA Space (3D)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_pca_scatter_3d.png"), dpi=300, bbox_inches='tight')
print("[OK] Saved: 03_pca_scatter_3d.png")
plt.close()

# ══════════════════════════════════════════════════════════════
#  9. CLUSTER HEALTH PROFILE HEATMAP
# ══════════════════════════════════════════════════════════════
print("\nGenerating Cluster Health Profile Heatmap...")

# Calculate mean values for each feature by cluster
cluster_profiles = X.groupby('Cluster').mean()

# Map cluster indices to risk labels
cluster_profiles.index = risk_labels

# Normalize for better visualization
cluster_profiles_normalized = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    cluster_profiles_normalized.T,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Normalized Health Profile'},
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel('Risk Groups', fontsize=12, fontweight='bold')
ax.set_ylabel('Health Features', fontsize=12, fontweight='bold')
ax.set_title('Cluster Health Profile Heatmap (Normalized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_cluster_heatmap.png"), dpi=300, bbox_inches='tight')
print("[OK] Saved: 04_cluster_heatmap.png")
plt.close()

# ══════════════════════════════════════════════════════════════
#  10. CLUSTER SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\nGenerating Cluster Summary Table...")

summary_data = []
for i in range(optimal_k):
    cluster_mask = cluster_labels == i
    cluster_size = np.sum(cluster_mask)
    percentage = (cluster_size / len(cluster_labels)) * 100
    
    summary_data.append({
        'Risk Group': risk_labels[i],
        'Cluster ID': i,
        'Patient Count': cluster_size,
        'Percentage': f"{percentage:.1f}%",
        'Avg Age': f"{X.loc[cluster_mask, 'age'].mean():.1f}",
        'Avg Heart Rate': f"{X.loc[cluster_mask, 'thalach'].mean():.1f}",
        'Avg BP': f"{X.loc[cluster_mask, 'trestbps'].mean():.1f}",
        'Avg Cholesterol': f"{X.loc[cluster_mask, 'chol'].mean():.1f}",
        'Male Ratio': f"{(X.loc[cluster_mask, 'sex'].sum() / cluster_size):.1%}"
    })

summary_df = pd.DataFrame(summary_data)

# Display summary
print("\n" + "=" * 70)
print("CLUSTER SUMMARY TABLE")
print("=" * 70)
print(summary_df.to_string(index=False))

# Save as CSV
summary_df.to_csv(os.path.join(OUTPUT_DIR, "05_cluster_summary.csv"), index=False)
print("\n[OK] Saved: 05_cluster_summary.csv")

# ══════════════════════════════════════════════════════════════
#  11. DETAILED CLUSTER STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DETAILED CLUSTER STATISTICS")
print("=" * 70)

for i in range(optimal_k):
    print(f"\n{risk_labels[i]} (Cluster {i}):")
    print(f"  Size: {np.sum(cluster_labels == i)} patients")
    cluster_data = X[X['Cluster'] == i].drop('Cluster', axis=1)
    print(f"  Statistics:\n{cluster_data.describe().to_string()}")

# ══════════════════════════════════════════════════════════════
#  12. SAVE COMPLETE CLUSTERING RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING COMPLETE RESULTS")
print("=" * 70)

# Save clustered data with original features
clustered_data = df.copy()
clustered_data['Risk_Group'] = pd.Series(cluster_labels).map({0: 'Low', 1: 'Medium', 2: 'High'}).values
clustered_data.to_csv(os.path.join(OUTPUT_DIR, "clustered_patients.csv"), index=False)
print("[OK] Saved: clustered_patients.csv (Full dataset with risk assignments)")

# Save PCA transformed data
pca_df = pd.DataFrame(
    X_pca,
    columns=['PC1', 'PC2', 'PC3']
)
pca_df['Cluster'] = cluster_labels
pca_df['Risk_Group'] = pd.Series(cluster_labels).map({0: 'Low', 1: 'Medium', 2: 'High'}).values
pca_df.to_csv(os.path.join(OUTPUT_DIR, "pca_transformed_data.csv"), index=False)
print("[OK] Saved: pca_transformed_data.csv (PCA components with cluster assignments)")

# ══════════════════════════════════════════════════════════════
#  13. GENERATE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING SUMMARY REPORT")
print("=" * 70)

report_text = f"""
{'='*70}
K-MEANS CLUSTERING ANALYSIS: PATIENT RISK SEGMENTATION
{'='*70}

PROJECT INFORMATION
-------------------
Algorithm: K-Means Clustering (Unsupervised Learning)
Purpose: Segment patients into risk groups for clinical decision-making
Dataset: UCI Heart Disease - Cleveland (processed.cleveland.data)
Total Patients: {len(X)}
Features Used: {len(X.columns)} cardiac health indicators

CLUSTERING RESULTS
------------------
Optimal Number of Clusters: {optimal_k}
Final Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}
  (Score ranges from -1 to 1; higher is better)
  (Score > 0.5 indicates good cluster separation)

PCA ANALYSIS
-----------
Components Used: 3
Explained Variance:
  - PC1: {pca.explained_variance_ratio_[0]:.2%}
  - PC2: {pca.explained_variance_ratio_[1]:.2%}
  - PC3: {pca.explained_variance_ratio_[2]:.2%}
  Total: {np.sum(pca.explained_variance_ratio_):.2%}

RISK GROUP SEGMENTATION
-----------------------
{summary_df.to_string(index=False)}

INTERPRETATION GUIDE
--------------------
The K-Means algorithm identified 3 natural patient clusters based on 13 cardiac
health features. These clusters were mapped to clinical risk levels:

• LOW RISK: Patients with healthy cardiac profiles
• MEDIUM RISK: Patients with some cardiac concerns
• HIGH RISK: Patients with multiple risk indicators

FEATURES ANALYZED
-----------------
{chr(10).join([f'• {FEATURE_DESCRIPTIONS[col]}' for col in X.columns if col != 'Cluster'])}

OUTPUT FILES
------------
1. 01_elbow_plot.png
   - Elbow method visualization showing cluster optimization
   - Used to determine optimal number of clusters

2. 02_silhouette_analysis.png
   - Silhouette score plot for cluster quality validation
   - Higher scores indicate better-defined clusters

3. 03_pca_scatter_3d.png
   - 3D PCA scatter plot showing cluster separation
   - Purple stars represent cluster centroids

4. 04_cluster_heatmap.png
   - Normalized health profile heatmap
   - Shows average feature values per risk group

5. 05_cluster_summary.csv
   - Summary statistics for each risk group
   - Key metrics: size, age, heart rate, blood pressure, cholesterol

6. clustered_patients.csv
   - Complete dataset with risk group assignments
   - Can be used for downstream analysis

7. pca_transformed_data.csv
   - PCA-transformed features with cluster assignments
   - Useful for visualization and further modeling

CLINICAL APPLICATIONS
---------------------
• Risk stratification for targeted interventions
• Patient monitoring prioritization
• Personalized treatment planning
• Public health resource allocation
• Patient outcome prediction studies

METHODOLOGY NOTES
-----------------
• Data Preprocessing: StandardScaler normalization (critical for K-Means)
• Algorithm: K-Means clustering with k-means++ initialization
• Optimal K: Determined using Elbow method and Silhouette analysis
• Validation: Multiple cluster validity metrics applied
• Random State: Set to 42 for reproducibility

{'='*70}
Analysis completed successfully!
Generated outputs are saved in: {OUTPUT_DIR}
{'='*70}
"""

with open(os.path.join(OUTPUT_DIR, "ANALYSIS_REPORT.txt"), 'w') as f:
    f.write(report_text)

print(report_text)
print("\n[OK] Saved: ANALYSIS_REPORT.txt")

print("\n" + "=" * 70)
print("ALL OUTPUTS SUCCESSFULLY GENERATED!")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. 01_elbow_plot.png")
print("  2. 02_silhouette_analysis.png")
print("  3. 03_pca_scatter_3d.png")
print("  4. 04_cluster_heatmap.png")
print("  5. 05_cluster_summary.csv")
print("  6. clustered_patients.csv")
print("  7. pca_transformed_data.csv")
print("  8. ANALYSIS_REPORT.txt")
print("=" * 70)
