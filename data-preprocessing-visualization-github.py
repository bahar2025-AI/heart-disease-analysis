"""
Advanced Data Preprocessing and Visualization
Author: [Your Name]
Date: March 2025

This script performs comprehensive data preprocessing and visualization on the UCI Heart Disease dataset.
Features include:
- Missing value handling
- Outlier detection and treatment
- Feature engineering
- Exploratory data analysis
- Interactive visualizations
- Dimensionality reduction
- Feature importance analysis
"""

# Install required packages
# Uncomment if you need to install these packages
# !pip install pandas numpy matplotlib seaborn plotly scikit-learn umap-learn yellowbrick missingno ydata-profiling

# --------------------- IMPORT LIBRARIES ---------------------
# Import basic libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import specialized visualization libraries
import missingno as msno
from ydata_profiling import ProfileReport

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# --------------------- DOWNLOAD AND LOAD DATA ---------------------
# Download the UCI Heart Disease dataset (Cleveland)
# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data -O heart_disease.csv

# Define column names (the original dataset doesn't have headers)
print("Loading dataset...")
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load the dataset into a pandas DataFrame
df = pd.read_csv('heart_disease.csv', header=None, names=column_names)

# Explore the data - basic overview
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

# --------------------- INITIAL DATA PREPROCESSING ---------------------
# Convert '?' values to NaN (missing values)
for col in df.columns:
    df[col] = df[col].replace('?', np.nan)

# Define numeric and categorical columns
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

# Convert numeric columns to appropriate types
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert categorical columns to numeric types
for col in categorical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nUpdated missing values after conversion:")
print(df.isnull().sum())

# --------------------- MISSING VALUE ANALYSIS ---------------------
print("\n========== MISSING VALUE ANALYSIS ==========")
# Visualize missing values with a matrix
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('Missing Value Matrix')
plt.tight_layout()
plt.show()

# Visualize missing values with a bar chart
plt.figure(figsize=(12, 6))
msno.bar(df)
plt.title('Missing Value Bar Chart')
plt.tight_layout()
plt.show()

# --------------------- MISSING VALUE IMPUTATION ---------------------
print("\nImputing missing values...")
# For numeric columns - use KNN imputation (more sophisticated approach)
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# For categorical columns - use mode imputation (most frequent value)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Missing values after imputation:")
print(df.isnull().sum())

# --------------------- OUTLIER DETECTION AND TREATMENT ---------------------
print("\n========== OUTLIER ANALYSIS ==========")
# Visualize outliers with boxplots
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Detect outliers using Isolation Forest algorithm
print("\nDetecting outliers with Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso_forest.fit_predict(df[numeric_cols])
outliers = df[df['outlier'] == -1]
print(f"Number of detected outliers: {len(outliers)}")

# Treat outliers using capping method (limit values to a defined range)
print("\nTreating outliers with capping method...")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)

# Remove outlier column as it's no longer needed
df = df.drop('outlier', axis=1)

# --------------------- FEATURE ENGINEERING ---------------------
print("\n========== FEATURE ENGINEERING ==========")
# Create age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], 
                        labels=['<40', '40-55', '55-70', '>70'])

# Create cholesterol level categories
df['chol_level'] = pd.cut(df['chol'], bins=[0, 200, 240, 500], 
                         labels=['Normal', 'Borderline', 'High'])

# Create blood pressure categories
df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 300], 
                          labels=['Normal', 'Prehypertension', 'Hypertension'])

# Simulate height and weight for BMI calculation (for demonstration)
# Note: These are not real values, just for demonstration purposes
np.random.seed(42)
df['height'] = np.random.normal(170, 10, size=len(df))  # cm
df['weight'] = np.random.normal(75, 15, size=len(df))   # kg
df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Calculate heart rate reserve
df['max_hr'] = 220 - df['age']  # Theoretical maximum heart rate
df['hr_reserve'] = df['max_hr'] - df['thalach']  # Heart rate reserve

# --------------------- FEATURE SCALING ---------------------
print("\nScaling numerical features...")
# Apply standard scaling (mean=0, std=1)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Display the results of feature engineering
print("\nResults of feature engineering:")
print(df[['age', 'age_group', 'chol', 'chol_level', 'trestbps', 
         'bp_category', 'bmi', 'bmi_category']].head(10))

# --------------------- EXPLORATORY DATA ANALYSIS ---------------------
print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Distribution of target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Heart Disease Classes')
plt.xlabel('Heart Disease (0 = No, 1-4 = Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Age distribution by heart disease
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='target', bins=20, multiple='stack')
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Cholesterol by heart disease
plt.figure(figsize=(12, 6))
sns.boxplot(x='target', y='chol', data=df)
plt.title('Cholesterol Levels by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Cholesterol')
plt.tight_layout()
plt.show()

# 4. Heart rate vs. age with disease status
plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='viridis')
plt.title('Maximum Heart Rate vs. Age')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate')
plt.tight_layout()
plt.show()

# 5. Gender distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='sex', hue='target', data=df)
plt.title('Heart Disease by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6. Correlation matrix
plt.figure(figsize=(14, 10))
corr_matrix = df[numeric_cols + ['target']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# --------------------- ADVANCED VISUALIZATIONS WITH PLOTLY ---------------------
print("\n========== ADVANCED VISUALIZATIONS ==========")

# 1. 3D Scatter plot of age, cholesterol, and heart rate
fig = px.scatter_3d(df, x='age', y='chol', z='thalach', color='target',
                   title='3D Scatter Plot: Age, Cholesterol, Heart Rate',
                   labels={'age': 'Age', 'chol': 'Cholesterol', 'thalach': 'Max Heart Rate'})
fig.show()

# 2. Parallel coordinates plot for multivariate analysis
fig = px.parallel_coordinates(df, 
                             color='target',
                             dimensions=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                             title='Parallel Coordinates Plot of Heart Disease Features')
fig.show()

# 3. Radar chart for different heart disease classes
fig = go.Figure()

for target_val in df['target'].unique():
    subset = df[df['target'] == target_val]
    mean_vals = subset[numeric_cols].mean()
    fig.add_trace(go.Scatterpolar(
        r=mean_vals.values,
        theta=mean_vals.index,
        fill='toself',
        name=f'Class {target_val}'
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )),
    showlegend=True,
    title='Radar Chart of Mean Values by Heart Disease Class'
)
fig.show()

# 4. Sunburst chart for hierarchical view
fig = px.sunburst(df, 
                 path=['sex', 'age_group', 'chol_level'], 
                 color='target',
                 title='Hierarchical View of Demographics and Heart Disease')
fig.show()

# --------------------- DIMENSIONALITY REDUCTION FOR VISUALIZATION ---------------------
print("\n========== DIMENSIONALITY REDUCTION ==========")

# Select features for dimensionality reduction
features_for_dr = numeric_cols

# 1. PCA (Principal Component Analysis)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[features_for_dr])
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
df_pca['target'] = df['target']

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca, palette='viridis')
plt.title('PCA: First Two Principal Components')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.tight_layout()
plt.show()

# 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df[features_for_dr])
df_tsne = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
df_tsne['target'] = df['target']

plt.figure(figsize=(12, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='target', data=df_tsne, palette='viridis')
plt.title('t-SNE Projection')
plt.tight_layout()
plt.show()

# 3. UMAP (Uniform Manifold Approximation and Projection)
reducer = umap.UMAP(random_state=42)
umap_result = reducer.fit_transform(df[features_for_dr])
df_umap = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
df_umap['target'] = df['target']

plt.figure(figsize=(12, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='target', data=df_umap, palette='viridis')
plt.title('UMAP Projection')
plt.tight_layout()
plt.show()

# --------------------- FEATURE IMPORTANCE ANALYSIS ---------------------
print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

# Prepare data for modeling
# Convert target to binary (0 = no disease, 1 = disease)
X = df[numeric_cols + ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']]
y = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Display feature importances as a table
print("\nFeature Importance Table:")
print(importances)

# --------------------- INTERACTIVE DASHBOARD WITH PLOTLY ---------------------
print("\n========== CREATING INTERACTIVE DASHBOARD ==========")

# Create a dashboard with multiple plots
dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Age vs. Heart Rate by Disease Status', 
                   'Cholesterol Distribution by Disease Status',
                   'Blood Pressure vs Age by Disease Status',
                   'Feature Importances'),
    specs=[[{'type': 'scatter'}, {'type': 'box'}],
           [{'type': 'scatter'}, {'type': 'bar'}]]
)

# Plot 1: Age vs. Heart Rate
for target_val in sorted(df['target'].unique()):
    subset = df[df['target'] == target_val]
    dashboard.add_trace(
        go.Scatter(
            x=subset['age'], 
            y=subset['thalach'],
            mode='markers',
            name=f'Class {target_val}',
            marker=dict(size=8)
        ),
        row=1, col=1
    )

# Plot 2: Cholesterol Distribution
for target_val in sorted(df['target'].unique()):
    subset = df[df['target'] == target_val]
    dashboard.add_trace(
        go.Box(
            y=subset['chol'],
            name=f'Class {target_val}'
        ),
        row=1, col=2
    )

# Plot 3: Blood Pressure vs Age
for target_val in sorted(df['target'].unique()):
    subset = df[df['target'] == target_val]
    dashboard.add_trace(
        go.Scatter(
            x=subset['age'],
            y=subset['trestbps'],
            mode='markers',
            name=f'Class {target_val}',
            marker=dict(size=8)
        ),
        row=2, col=1
    )

# Plot 4: Feature Importances
dashboard.add_trace(
    go.Bar(
        x=importances['Importance'][:8],
        y=importances['Feature'][:8],
        orientation='h'
    ),
    row=2, col=2
)

# Update dashboard layout
dashboard.update_layout(
    height=800,
    width=1200,
    title_text='Heart Disease Analysis Dashboard',
    showlegend=False
)

dashboard.show()

# --------------------- COMPREHENSIVE REPORT GENERATION ---------------------
# Uncomment the following lines to generate a comprehensive report
# Note: This can take some time to complete
"""
print("\n========== GENERATING COMPREHENSIVE REPORT ==========")
profile = ProfileReport(df, title="Heart Disease Dataset Profiling Report", minimal=True)
profile.to_file("heart_disease_profile_report.html")
"""

# --------------------- SAVE PROCESSED DATA ---------------------
print("\n========== SAVING PROCESSED DATA ==========")
df.to_csv('processed_heart_disease.csv', index=False)
df_scaled.to_csv('scaled_heart_disease.csv', index=False)
print("Processed data saved to CSV files.")

# --------------------- SUMMARY AND CONCLUSIONS ---------------------
print("\n========== SUMMARY ==========")
print("Data preprocessing and visualization complete!")
print("Key findings:")
print("1. The dataset contains information on", len(df), "patients.")
print("2. Top features correlated with heart disease:")
for i, row in importances.head(5).iterrows():
    print(f"   - {row['Feature']}: {row['Importance']:.4f}")
print("3. Visualizations displayed and saved.")
print("4. Processed datasets saved as CSV files.")

# Additional analysis: Heart disease by age group and gender
plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1)
sns.countplot(x='age_group', hue='target', data=df, ax=ax1)
plt.title('Heart Disease by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Heart Disease')

ax2 = plt.subplot(1, 2, 2)
# Convert gender to text for better readability
df['gender'] = df['sex'].map({0: 'Female', 1: 'Male'})
sns.countplot(x='gender', hue='target', data=df, ax=ax2)
plt.title('Heart Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Heart Disease')

plt.tight_layout()
plt.show()

print("\nThank you for using this advanced data preprocessing and visualization script!")
