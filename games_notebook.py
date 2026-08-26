import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df=pd.read_csv(r"C:\Users\pc\Downloads\Prolect\android_games_eda_ready.csv")

df

df.shape

df.info()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

df.dtypes

df.isnull().sum()

cat_cols=df.select_dtypes(include='object').columns
num_cols=df.select_dtypes(include=np.number).columns
df[cat_cols]=df[cat_cols].fillna('Unknown').astype(str)
df[num_cols]=df[num_cols].fillna(df[num_cols].median())

df.isnull().sum()

df['is_hit_game'].unique()

df['is_hit_game'].value_counts()

plt.bar(['0','1'], [9800,200],color='red',facecolor='blue',edgecolor='black',hatch='//')
plt.title('Distribution of Hit Games')
plt.xlabel('Is hit game')
plt.ylabel('Number of games')
plt.show()

corr=df.corr(numeric_only=True)

plt.figure(figsize=(25,25))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm", linewidth=1.5)
plt.show()

cols_to_drop = [
    'game_id', 'game_name', 'package_name', 'developer_name',
    'release_date', 'soft_launch_date', 'last_update_date',
    'featured_start_date', 'featured_end_date', 'row_checksum_id'
]
df_prep = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df.shape

x=df_prep.drop(columns=['is_hit_game'])
y=df_prep['is_hit_game']

x_encoded=pd.get_dummies(x,drop_first=True)

x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

print(f"Encoded features count: {x_encoded.shape[1]}")
print(f"Training set size: {x_train.shape[0]} samples")
print(f"Testing set size: {x_test.shape[0]} samples")

pca=PCA(n_components=0.95,random_state=42)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca=pca.transform(x_test_scaled)

print(f"Original feature dimensions: {x_train_scaled.shape[1]}")
print(f"Reduced PCA feature dimensions: {x_train_pca.shape[1]}")

knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled, y_train)
y_pred_knn=knn_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_knn))

cm_knn=confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: KNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

dt_model=DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
dt_model.fit(x_train_scaled, y_train)
y_pred_dt=dt_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

nb_model=GaussianNB()
nb_model.fit(x_train_pca, y_train)
y_pred_nb=nb_model.predict(x_test_pca)

print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

svm_model=SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(x_train_scaled, y_train)
y_pred_svm=svm_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: Support Vector Machine')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

rf_model=RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(x_train_scaled, y_train)
y_pred_rf=rf_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

gb_model=GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gb_model.fit(x_train_scaled, y_train)
y_pred_gb=gb_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_gb))

cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: Gradient Boosting')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from xgboost import XGBClassifier
xg_model=XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=3,max_leaves=5,random_state=42, scale_pos_weight=7000/200)
xg_model.fit(x_train_scaled, y_train)
y_pred_xg=xg_model.predict(x_test_scaled)

print(classification_report(y_test, y_pred_xg))

cm_xgb = confusion_matrix(y_test, y_pred_xg)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Hit', 'Hit'], yticklabels=['Non-Hit', 'Hit'])
plt.title('Confusion Matrix: XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

models={
    'Gradient Boosting': gb_model,
    'XGBoost': xg_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'SVM': svm_model,
    'KNN': knn_model,
    'Naive Bayes': nb_model
}
scores={}
for name, model in models.items():
    x_test_data=x_test_pca if name == 'Naive Bayes' else x_test_scaled
    preds=model.predict(x_test_data)
    scores[name]=f1_score(y_test, preds)
plt.figure(figsize=(8, 4))
plt.barh(list(scores.keys()), list(scores.values()), color='skyblue')
plt.title('F1-Score Comparison')
plt.xlabel('F1-Score')
plt.tight_layout()
plt.show()