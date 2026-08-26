import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

df = pd.read_csv(r"C:\Users\pc\Downloads\Prolect\android_games_eda_ready.csv")
df.drop_duplicates(inplace=True)

cols_to_drop = [
    'game_id', 'game_name', 'package_name', 'developer_name',
    'release_date', 'soft_launch_date', 'last_update_date',
    'featured_start_date', 'featured_end_date', 'row_checksum_id'
]
df_prep = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

X = df_prep.drop(columns=['is_hit_game'])
y = df_prep['is_hit_game']

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

models = {
    'KNN': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42))
    ]),
    'Naive Bayes (with PCA)': Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('classifier', GaussianNB())
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, max_leaves=5, random_state=42, scale_pos_weight=7000/200))
    ])
}

results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    results[name] = f1_score(y_test, pipeline.predict(X_test))

for model_name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{model_name:25s}: {score:.4f}")