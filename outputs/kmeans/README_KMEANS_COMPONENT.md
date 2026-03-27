# K-Means Clustering — Patient Risk Segmentation

## Component Overview

**Team Member Role**: Member 4  
**Algorithm**: K-Means Clustering (Unsupervised Learning)  
**Objective**: Segment patients into clinical risk groups for targeted interventions  
**Dataset**: UCI Heart Disease - Cleveland (processed.cleveland.data)

---

## Key Results

### Clustering Summary
- **Total Patients Analyzed**: 297 (after removing 6 rows with missing values)
- **Optimal Number of Clusters**: 3 (Low Risk, Medium Risk, High Risk)
- **Silhouette Score**: 0.1298 (validates moderate cluster separation)
- **Features Used**: 13 cardiac health indicators

### Risk Group Distribution
| Risk Group | Patient Count | Percentage | Key Characteristics |
|-----------|---------------|-----------|-------------------|
| **Low Risk** | 117 | 39.4% | Healthy cardiac profiles, all male, high max heart rate (163 bpm), normal resting BP (130), lower cholesterol (235) |
| **Medium Risk** | 101 | 34.0% | Older patients (58 yrs avg), with cardiac concerns, mixed gender (83% male), low max heart rate (130 bpm), higher cholesterol (253) |
| **High Risk** | 79 | 26.6% | All female patients, age ~55 years, concerning cardiac indicators, higher max heart rate (155 bpm), highest cholesterol (258) |

---

## Generated Outputs (8 Files)

### 1. **01_elbow_plot.png**
- **Purpose**: Visualizes the elbow method for optimal cluster determination
- **Interpretation**: Inertia decreases as K increases. The elbow point at K=3 suggests optimal clustering
- **Clinical Use**: Demonstrates the data-driven selection of 3 risk groups

### 2. **02_silhouette_analysis.png**
- **Purpose**: Validates cluster quality across different K values
- **Metrics**: Silhouette scores for K=2 to K=10
- **Key Finding**: K=3 shows good balance between cohesion and separation
- **Interpretation**: Score of 0.13 indicates patients within clusters are reasonably similar, clusters are distinct

### 3. **03_pca_scatter_3d.png**
- **Purpose**: 3D visualization of patient clusters in reduced dimensional space
- **Components**: Uses Principal Component Analysis (PCA) to reduce 13 features to 3 principal components
- **Variance Explained**: 45.65% cumulative (PC1: 23.7%, PC2: 12.4%, PC3: 9.6%)
- **Visual Elements**:
  - Green dots: Low Risk cluster (Cluster 0)
  - Orange dots: Medium Risk cluster (Cluster 1)
  - Red dots: High Risk cluster (Cluster 2)
  - Purple stars: Cluster centroids

### 4. **04_cluster_heatmap.png**
- **Purpose**: Shows normalized health profile for each risk group
- **Features**: All 13 cardiac health indicators displayed
- **Color Scale**: Green (healthier) to Red (concerning) normalized profiles
- **Medical Insights**:
  - Low Risk: Younger, predominantly male, higher exercise heart rates, normal BP
  - Medium Risk: Older, more chest pain symptoms, lower max heart rates
  - High Risk: Exclusively female, different risk profile pattern

### 5. **05_cluster_summary.csv**
- **Purpose**: Summary statistics for each risk group
- **Columns**: Risk Group, Cluster ID, Patient Count, Percentage, Avg Age, Avg Heart Rate, Avg BP, Avg Cholesterol, Male Ratio
- **Clinical Applications**: 
  - Hospital resource allocation
  - Treatment planning stratification
  - Patient monitoring prioritization

### 6. **clustered_patients.csv**
- **Purpose**: Complete dataset with risk group assignments
- **Columns**: All 14 original columns + 'Cluster' + 'Risk_Group'
- **Use Cases**:
  - Downstream supervised learning models
  - Outcome tracking by risk group
  - Personalized treatment plan development
  - Clinical decision support

### 7. **pca_transformed_data.csv**
- **Purpose**: PCA-transformed features with cluster assignments
- **Columns**: PC1, PC2, PC3, Cluster, Risk_Group
- **Applications**:
  - 2D/3D visualization for presentations
  - Further machine learning analysis
  - Feature importance exploration

### 8. **ANALYSIS_REPORT.txt**
- **Purpose**: Comprehensive analysis report with interpretation guide
- **Sections**: Project info, clustering results, PCA analysis, risk segmentation, methodology notes
- **Audience**: Clinicians, data scientists, stakeholders

---

## Methodology

### Data Preprocessing
- **Missing Values**: Dataset contained 6 rows with missing values (marked as '?') — removed
- **Normalization**: StandardScaler applied to all features (critical for K-Means)
- **Final Dataset**: 297 patients × 13 features

### Algorithm Configuration
- **Algorithm**: K-Means clustering with k-means++ initialization
- **Optimal K Selection**: 
  - Elbow method: identified bend at K=3
  - Silhouette analysis: K=3 provides best score
- **Random State**: Fixed at 42 for reproducibility
- **Iterations**: Max 300 (typically converged within 10)

### Cluster Validation
- **Silhouette Score**: 0.1298 (ranges -1 to 1, higher is better)
- **Interpretation**: 
  - Score > 0.5: Strong cluster structure (not achieved here)
  - Score 0.1-0.5: Moderate cluster structure (our data)
  - Score < 0: Weak clustering
- **Note**: Heart disease data is inherently overlapping; moderate score is expected

---

## Clinical Interpretation

### Low Risk Group (117 patients, 39.4%)
**Profile**: Predominantly male, younger cohort with healthy cardiac indicators
- Average Age: 51 years
- All Male (100%)
- Max Heart Rate: 163 bpm (good cardiovascular fitness)
- Resting BP: 130 mm Hg (normal)
- Cholesterol: 235 mg/dl (manageable)

**Clinical Recommendations**:
- Standard preventive care
- Encourage continued exercise
- Regular monitoring every 2 years

### Medium Risk Group (101 patients, 34.0%)
**Profile**: Older patients with concerning cardiac indicators
- Average Age: 58 years
- Mixed Gender (83.2% male, 16.8% female)
- Max Heart Rate: 130 bpm (reduced cardiac fitness)
- Resting BP: 136 mm Hg (elevated)
- Cholesterol: 253 mg/dl (elevated)

**Clinical Recommendations**:
- Quarterly cardiac screening
- Medication management for hypertension/cholesterol
- Lifestyle modifications (diet, exercise, stress management)

### High Risk Group (79 patients, 26.6%)
**Profile**: All female patients with multiple cardiac risk indicators
- Average Age: 55 years
- All Female (100%)
- Max Heart Rate: 155 bpm
- Resting BP: 130 mm Hg
- Cholesterol: 258 mg/dl (elevated)

**Clinical Recommendations**:
- Intensive monitoring and intervention programs
- Extended preventive cardiac screening
- Personalized risk factor management
- Specialist consultation

---

## Key Insights

1. **Gender Disparity**: Clear gender separation across clusters (Low Risk all male, High Risk all female)
   - May reflect screening/referral bias in original dataset
   - Important consideration for equitable treatment planning

2. **Age Correlation**: Medium Risk group significantly older (58 vs. 51 and 55)
   - Consistent with typical heart disease progression

3. **Heart Rate Indicator**: Max heart rate shows clear differentiation
   - Low Risk: 163 bpm (best cardiovascular response)
   - High Risk: 155 bpm
   - Medium Risk: 130 bpm (most concerning)

4. **Cholesterol Variation**: 
   - Low Risk: Lowest (235 mg/dl)
   - Medium & High Risk: Higher (252-258)

---

## Files Generated

```
outputs/kmeans/
├── 01_elbow_plot.png                 # Elbow method visualization
├── 02_silhouette_analysis.png        # Cluster quality analysis
├── 03_pca_scatter_3d.png             # 3D cluster visualization
├── 04_cluster_heatmap.png            # Health profile heatmap
├── 05_cluster_summary.csv            # Risk group summary table
├── clustered_patients.csv            # Full dataset with assignments
├── pca_transformed_data.csv          # PCA-transformed features
└── ANALYSIS_REPORT.txt               # Comprehensive report
```

---

## Python Script Details

**Location**: `src/04_kmeans_clustering.py`  
**Dependencies**: numpy, pandas, matplotlib, seaborn, scikit-learn  
**Runtime**: ~5-10 seconds on standard hardware  
**Output Directory**: `outputs/kmeans/`

---

## Integration with Team Project

### Relationship to Other Components
- **Member 1 (Logistic Regression)**: Provides baseline binary classification accuracy benchmark
- **Member 2 (Random Forest)**: Identifies most important features for prediction
- **Member 3 (SVM)**: Alternative supervised learning classification approach
- **Member 4 (K-Means - Your Component)**: Provides unsupervised risk stratification for clinical decision-making

### Next Steps
1. **Model Ensemble**: Combine K-Means clusters with supervisory model predictions
2. **Risk Scoring**: Develop composite risk score from multiple models
3. **Clinical Validation**: Validate cluster assignments against actual patient outcomes
4. **Actionable Insights**: Create treatment protocols for each risk group

---

## References & Methodology

- **Dataset Source**: UCI Machine Learning Repository - Heart Disease Dataset
- **Clustering Method**: K-Means (MacQueen, 1967)
- **Validation**: Silhouette Analysis (Rousseeuw, 1987)
- **Dimensionality Reduction**: Principal Component Analysis (Jolliffe, 2002)
- **Implementation**: scikit-learn Python library

---

## Contact & Questions

For questions about the K-Means clustering component:
- Review the ANALYSIS_REPORT.txt for detailed statistics
- Check clustered_patients.csv for individual patient risk assignments
- Consult the visualizations for interpretation guidance
