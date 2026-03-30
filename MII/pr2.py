import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv('ICU_Patient_Monitoring_Mortality_Prediction_15000.csv')

print("Размер датасета:", df.shape)
print("Количество строк:", df.shape[0])
print("Количество столбцов:", df.shape[1])

print("\nТипы данных:")
print(df.dtypes)

print("\nПропуски в данных (сумма по каждому столбцу):")
print(df.isnull().sum())
print("\nОбщее количество пропусков:", df.isnull().sum().sum())

print("\nМинимальные значения числовых признаков:")
print(df.select_dtypes(include=[np.number]).min())
print("\nМаксимальные значения числовых признаков:")
print(df.select_dtypes(include=[np.number]).max())

print("\nРаспределение mortality_label:")
class_dist = df['mortality_label'].value_counts()
print(class_dist)
mortality_rate = df['mortality_label'].mean() * 100
print(f"Доля умерших пациентов: {mortality_rate:.2f}%")
survival_rate = (1 - df['mortality_label'].mean()) * 100
print(f"Доля выживших пациентов: {survival_rate:.2f}%")

fig1 = px.histogram(df, x="age", color="mortality_label", nbins=50,
                     title="Распределение возраста пациентов по исходам",
                     labels={"age": "Возраст (годы)", "mortality_label": "Исход", "0": "Выжил", "1": "Умер"},
                     opacity=0.7)
fig1.show()

fig3 = px.box(df, x="mortality_label", y="sofa_score",
              title="Распределение SOFA score для выживших и умерших пациентов",
              labels={"mortality_label": "Исход", "sofa_score": "SOFA score", "0": "Выжил", "1": "Умер"},
              points="outliers")
fig3.show()

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X = df.drop(['mortality_label', 'patient_id'], axis=1, errors='ignore')
y = df['mortality_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(constant=0, random_state=42)
dummy.fit(X_train, y_train)

y_pred_proba_dummy = dummy.predict_proba(X_test)[:, 1]
roc_auc_dummy = roc_auc_score(y_test, y_pred_proba_dummy)
print(f"ROC-AUC константного предсказания: {roc_auc_dummy:.4f}")

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Числовые признаки:", numerical_cols)
print("Категориальные признаки:", categorical_cols)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

# Обучение
baseline_pipeline.fit(X_train, y_train)

y_pred_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC логистической регрессии: {roc_auc:.4f}")

import optuna
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold


def objective(trial, X_train, y_train):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_iter': trial.suggest_int('max_iter', 200, 800, step=25),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 0.5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 130, 500),
    }
    clf = HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        **params,
    )
    pipeline = Pipeline(
        steps=[('preprocessor', clone(preprocessor)), ('classifier', clf)]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, y_pred_proba))

    return float(np.mean(cv_scores))


study = optuna.create_study(
    direction='maximize', study_name='енвлдедлегнрд'
)
study.optimize(
    lambda trial: objective(trial, X_train, y_train),
    n_trials=50,
    show_progress_bar=True,
)

print("\nЛучшие гиперпараметры:")
print(study.best_params)
print(f"Лучшее значение ROC-AUC на кросс-валидации: {study.best_value:.4f}")

best_clf = HistGradientBoostingClassifier(
    random_state=42,
    class_weight='balanced',
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=25,
    **study.best_params,
)
final_pipeline = Pipeline(
    steps=[('preprocessor', clone(preprocessor)), ('classifier', best_clf)]
)
final_pipeline.fit(X_train, y_train)

y_pred_proba_final = final_pipeline.predict_proba(X_test)[:, 1]
roc_auc_final = roc_auc_score(y_test, y_pred_proba_final)

print(f"ROC-AUC константного предсказания: {roc_auc_dummy:.4f}")
print(f"ROC-AUC бейзлайна (LogReg): {roc_auc:.4f}")
print(f"ROC-AUC сложной модели (HistGradientBoosting + Optuna): {roc_auc_final:.4f}")

# Небольшой анализ
improvement = (roc_auc_final - roc_auc) / roc_auc * 100
print(f"\nУлучшение относительно бейзлайна: {improvement:.1f}%")

import shap
import matplotlib.pyplot as plt

print("Расчёт SHAP значений (может занять некоторое время)...")

X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
pre = final_pipeline.named_steps['preprocessor']
clf = final_pipeline.named_steps['classifier']
X_sample_t = pre.transform(X_sample)
feature_names = list(pre.get_feature_names_out())

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample_t)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Визуализация 1: summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample_t, feature_names=feature_names)
plt.title("SHAP Summary Plot – влияние признаков на прогноз смертности")
plt.tight_layout()
plt.show()

# Визуализация 2: bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample_t, feature_names=feature_names, plot_type="bar")
plt.title("Важность признаков (среднее абсолютное SHAP значение)")
plt.tight_layout()
plt.show()

print("\nИнтерпретация завершена.")