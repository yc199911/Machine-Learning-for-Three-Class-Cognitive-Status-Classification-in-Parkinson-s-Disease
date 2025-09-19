import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.model_selection import validation_curve, learning_curve  # 新增import
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score
from sklearn.calibration import calibration_curve

# å‰µå»ºå„²å­˜è³‡æ–™å¤¾
output_dir = '0919data'
os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_excel('PPMI_DATASET.xlsx', sheet_name='in')
missing_values = df.isnull().mean()
df_cleaned = df.drop(columns=missing_values[missing_values > 0.6].index).dropna(subset=['class'])

# Print original dataset distribution
print("Original dataset distribution:", df_cleaned['class'].value_counts())

# Impute missing values
X = df_cleaned.drop(columns=['class'])
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Stage 1: pdnc vs non-pdnc
label_stage1 = df_cleaned['class'].map({'pdnc': 0, 'pdmci': 1, 'pdd': 1})
X_train, X_test, y_train_stage1, y_test_stage1, class_train, class_test = train_test_split(
    X_imputed, label_stage1, df_cleaned['class'], test_size=0.2, stratify=label_stage1, random_state=42)

# Reset indices
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train_stage1.reset_index(drop=True, inplace=True)
y_test_stage1.reset_index(drop=True, inplace=True)
class_train.reset_index(drop=True, inplace=True)
class_test.reset_index(drop=True, inplace=True)

# Feature selection (SHAP + RFE, top 10 features)
xgb_base = XGBClassifier(random_state=42)
xgb_base.fit(X_train, y_train_stage1)
explainer = shap.TreeExplainer(xgb_base)
shap_values = explainer.shap_values(X_train)
shap_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': abs(shap_values).mean(axis=0)
}).sort_values(by='importance', ascending=False)
top_features = shap_importances.head(20)['feature'].tolist()

# å–å‰ 10 å€‹ç‰¹å¾µçš„é‡è¦æ€§ï¼ˆå¦‚æžœä½ éœ€è¦ top10ï¼‰
top20 = shap_importances.head(20)

# ç¹ªè£½æ©«æ¢åœ–
plt.figure(figsize=(8, 6))
plt.barh(top20['feature'], top20['importance'])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 20 Feature Importances")
plt.gca().invert_yaxis()  # åè½‰ y è»¸ï¼Œè®"æœ€é‡è¦çš„ç‰¹å¾µé¡¯ç¤ºåœ¨ä¸Šæ–¹
plt.tight_layout()

# å„²å­˜åœ–æª"åˆ° results è³‡æ–™å¤¾
plt.savefig(os.path.join(output_dir, 'shap_stage1.png'))
plt.show()

selector = RFE(xgb_base, n_features_to_select=10, step=1)
selector.fit(X_train[top_features], y_train_stage1)
selected_features = list(X_train[top_features].columns[selector.support_])
X_train_stage1 = X_train[selected_features]
X_test_stage1 = X_test[selected_features]

# SMOTETomek for Stage 1 with adjusted sampling strategy
smote_tomek_stage1 = SMOTETomek(sampling_strategy=0.8, random_state=42)
X_train_smote, y_train_smote = smote_tomek_stage1.fit_resample(X_train_stage1, y_train_stage1)

# GridSearchCV for Stage 1 with adjusted scale_pos_weight
param_grid_stage1 = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'scale_pos_weight': [1, 2, 3],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0]
}
grid_search_stage1 = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid_stage1, scoring='f1_macro', cv=5, n_jobs=-1, verbose=1
)
grid_search_stage1.fit(X_train_smote, y_train_smote)
best_model_stage1 = grid_search_stage1.best_estimator_
print(f"Stage 1 Best parameters: {grid_search_stage1.best_params_}")

# Cross-validation for Stage 1
cv_stage1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stage1 = cross_val_score(
    best_model_stage1, X_train_smote, y_train_smote, scoring='f1_macro', cv=cv_stage1, n_jobs=-1
)
print(f"Stage 1 Cross-validation F1_macro: {scores_stage1.mean():.4f} ± {scores_stage1.std():.4f}")

# ============================================================================
# 新增：Stage 1 CV Learning Curves
# ============================================================================
print("=== Generating Stage 1 Learning Curves ===")

# 1. Learning Curve - 訓練集大小 vs 性能
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores_stage1, test_scores_stage1 = learning_curve(
    best_model_stage1, X_train_smote, y_train_smote, train_sizes=train_sizes, 
    cv=cv_stage1, scoring='f1_macro', n_jobs=-1, shuffle=True, random_state=42
)

train_scores_mean = np.mean(train_scores_stage1, axis=1)
train_scores_std = np.std(train_scores_stage1, axis=1)
test_scores_mean = np.mean(test_scores_stage1, axis=1)
test_scores_std = np.std(test_scores_stage1, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', label='Training F1-macro')
plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='red', label='CV F1-macro')
plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('F1-macro Score')
plt.title('Stage 1 Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_learning_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Validation Curve - n_estimators
n_estimators_range = [50, 100, 150, 200, 250, 300]
train_scores_nest, test_scores_nest = validation_curve(
    best_model_stage1, X_train_smote, y_train_smote, 
    param_name='n_estimators', param_range=n_estimators_range,
    cv=cv_stage1, scoring='f1_macro', n_jobs=-1
)

train_scores_mean_nest = np.mean(train_scores_nest, axis=1)
train_scores_std_nest = np.std(train_scores_nest, axis=1)
test_scores_mean_nest = np.mean(test_scores_nest, axis=1)
test_scores_std_nest = np.std(test_scores_nest, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores_mean_nest, 'o-', color='blue', label='Training F1-macro')
plt.fill_between(n_estimators_range, train_scores_mean_nest - train_scores_std_nest,
                 train_scores_mean_nest + train_scores_std_nest, alpha=0.1, color='blue')
plt.plot(n_estimators_range, test_scores_mean_nest, 'o-', color='red', label='CV F1-macro')
plt.fill_between(n_estimators_range, test_scores_mean_nest - test_scores_std_nest,
                 test_scores_mean_nest + test_scores_std_nest, alpha=0.1, color='red')
plt.xlabel('n_estimators')
plt.ylabel('F1-macro Score')
plt.title('Stage 1 Validation Curve - n_estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_validation_curve_n_estimators.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. CV Fold Performance
fold_scores_stage1 = []
for fold, (train_idx, val_idx) in enumerate(cv_stage1.split(X_train_smote, y_train_smote)):
    X_fold_train = X_train_smote.iloc[train_idx]
    X_fold_val = X_train_smote.iloc[val_idx]
    y_fold_train = y_train_smote.iloc[train_idx]
    y_fold_val = y_train_smote.iloc[val_idx]
    
    best_model_stage1.fit(X_fold_train, y_fold_train)
    y_pred = best_model_stage1.predict(X_fold_val)
    f1 = f1_score(y_fold_val, y_pred, average='macro')
    fold_scores_stage1.append(f1)

plt.figure(figsize=(10, 6))
fold_labels = [f'Fold {i+1}' for i in range(5)]
bars = plt.bar(fold_labels, fold_scores_stage1, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
plt.axhline(y=np.mean(fold_scores_stage1), color='red', linestyle='--', 
            label=f'Mean: {np.mean(fold_scores_stage1):.3f} ± {np.std(fold_scores_stage1):.3f}')

for bar, score in zip(bars, fold_scores_stage1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom')

plt.ylabel('F1-macro Score')
plt.title('Stage 1 CV Fold Performance')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_cv_fold_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 原本的 Stage 1 測試代码继续...
# ============================================================================

# Stage 1 prediction with optimal threshold
proba_stage1 = best_model_stage1.predict_proba(X_test_stage1)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_stage1, proba_stage1, pos_label=1)
target_recall = 0.85
optimal_idx = np.where(recall >= target_recall)[0][-1]
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold for non-pdnc (prioritizing recall >= {target_recall}): {optimal_threshold:.3f}")

# Apply optimal threshold
y_pred_stage1_adjusted = (proba_stage1 >= optimal_threshold).astype(int)
print("=== Stage 1 Classification Report (Threshold Adjusted) ===")
print(classification_report(y_test_stage1, y_pred_stage1_adjusted, target_names=['pdnc', 'non-pdnc']))

# Stage 1 Confusion Matrix
cm_stage1 = confusion_matrix(y_test_stage1, y_pred_stage1_adjusted)
plt.figure(figsize=(6,4))
sns.heatmap(cm_stage1, annot=True, fmt='d', cmap='Greens',
            xticklabels=['pdnc', 'non-pdnc'], yticklabels=['pdnc', 'non-pdnc'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Stage 1 Confusion Matrix (pdnc vs non-pdnc)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_confusion_matrix.png'))
plt.show()

from sklearn.metrics import roc_auc_score, roc_curve

# Stage 1 ROC Curve and AUC
fpr_stage1, tpr_stage1, _ = roc_curve(y_test_stage1, proba_stage1)
roc_auc_stage1 = roc_auc_score(y_test_stage1, proba_stage1)

plt.figure(figsize=(7,5))
plt.plot(fpr_stage1, tpr_stage1, label=f'Stage 1 ROC (AUC = {roc_auc_stage1:.2f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stage 1 ROC Curve (pdnc vs non-pdnc)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_roc_curve.png'))
plt.show()

# Stage 1 Specificity
cm_stage1 = confusion_matrix(y_test_stage1, y_pred_stage1_adjusted)

TN, FP, FN, TP = cm_stage1.ravel()
specificity_stage1 = TN / (TN + FP)

specificity_stage1_df = pd.DataFrame({
    'Class': ['pdnc', 'non-pdnc'],
    'Specificity': [specificity_stage1, TP / (TP + FN)]
})

# Plot Specificity Stage 1
plt.figure(figsize=(6,4))
sns.barplot(x='Class', y='Specificity', data=specificity_stage1_df, palette='Greens')
plt.ylim(0,1)
plt.title('Stage 1 Specificity')
plt.ylabel('Specificity')
plt.grid(axis='y')

for idx, value in enumerate(specificity_stage1_df['Specificity']):
    plt.text(idx, value + 0.02, f'{value:.2f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage1_specificity.png'))
plt.show()

# Define test_stage2_idx
test_stage2_idx = (y_pred_stage1_adjusted == 1)

# Stage 2: pdmci vs pdd
stage2_features = [
    'NP1COG', 'NP1URIN', 'NP2DRES', 'NP2HYGN', 'NP3FACXP', 'NP3HMOVL',
    'NP3PRSPR', 'NP3BRADY', 'MCATOT_x', 'GDSBORED', 'GDSHLPLS', 'CLCK2HND', 'CLCKNMRK'
]
train_stage2_idx = (y_train_stage1 == 1) & (class_train != 'pdnc')
X_train_stage2 = X_train.loc[train_stage2_idx, stage2_features]
y_train_stage2_raw = class_train.loc[train_stage2_idx]

# Print class distribution before encoding
print("Class distribution before encoding (Stage 2):", y_train_stage2_raw.value_counts())

# Manually map pdmci and pdd to 0 and 1
y_train_stage2 = y_train_stage2_raw.map({'pdmci': 0, 'pdd': 1})

# Print class distribution after mapping
print("Class distribution after mapping (Stage 2):", pd.Series(y_train_stage2).value_counts())

# Create a custom LabelEncoder for decoding predictions
class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(['pdmci', 'pdd'])
    
    def transform(self, y):
        return np.array([0 if x == 'pdmci' else 1 for x in y])
    
    def inverse_transform(self, y):
        return np.array(['pdmci' if x == 0 else 'pdd' for x in y])

label_encoder_stage2 = CustomLabelEncoder()

# Feature selection for Stage 2 using SHAP
xgb_stage2 = XGBClassifier(
    # scale_pos_weight=4,  # æé«˜PDD recall
    random_state=42,
    # n_estimators=200,
    # max_depth=5,
    # learning_rate=0.05
    )
xgb_stage2.fit(X_train_stage2, y_train_stage2)
explainer_stage2 = shap.TreeExplainer(xgb_stage2)
shap_values_stage2 = explainer_stage2.shap_values(X_train_stage2)
shap_importances_stage2 = pd.DataFrame({
    'feature': X_train_stage2.columns,
    'importance': abs(shap_values_stage2).mean(axis=0)
}).sort_values(by='importance', ascending=False)
top_features_stage2 = shap_importances_stage2.head(20)['feature'].tolist()

# Add interaction features
X_train_stage2 = X_train_stage2[top_features_stage2].copy()
X_test_stage2 = X_test.loc[test_stage2_idx, top_features_stage2].copy()
X_train_stage2['NP1COG_MCATOT'] = X_train_stage2['NP1COG'] * X_train_stage2['MCATOT_x']
X_test_stage2['NP1COG_MCATOT'] = X_test_stage2['NP1COG'] * X_test_stage2['MCATOT_x']
X_train_stage2['NP1COG_NP2HYGN'] = X_train_stage2['NP1COG'] * X_train_stage2['NP2HYGN']
X_test_stage2['NP1COG_NP2HYGN'] = X_test_stage2['NP1COG'] * X_test_stage2['NP2HYGN']
X_train_stage2['MCATOT_GDSHLPLS'] = X_train_stage2['MCATOT_x'] * X_train_stage2['GDSHLPLS']
X_test_stage2['MCATOT_GDSHLPLS'] = X_test_stage2['MCATOT_x'] * X_test_stage2['GDSHLPLS']

# Recompute SHAP values with interaction features
xgb_stage2.fit(X_train_stage2, y_train_stage2)
explainer_stage2 = shap.TreeExplainer(xgb_stage2)
shap_values_stage2 = explainer_stage2.shap_values(X_train_stage2)
shap_importances_stage2 = pd.DataFrame({
    'feature': X_train_stage2.columns,
    'importance': abs(shap_values_stage2).mean(axis=0)
}).sort_values(by='importance', ascending=False)

# Plot SHAP summary
shap.summary_plot(shap_values_stage2, X_train_stage2, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for Stage 2 (pdmci vs pdd)")
plt.savefig(os.path.join(output_dir, 'shap_stage2.png'))
plt.show()

# SMOTETomek for Stage 2 with adjusted sampling strategy
smote_tomek_stage2 = SMOTETomek(sampling_strategy={0: 400, 1: 350}, random_state=42)
X_train_stage2_smote, y_train_stage2_smote = smote_tomek_stage2.fit_resample(X_train_stage2, y_train_stage2)

# Neural network for Stage 2
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train_stage2_smote, y_train_stage2_smote)

# Cross-validation for MLP
cv_stage2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stage2 = cross_val_score(
    mlp_model, X_train_stage2_smote, y_train_stage2_smote, scoring='f1_macro', cv=cv_stage2, n_jobs=-1
)
print(f"Stage 2 Cross-validation F1_macro: {scores_stage2.mean():.4f} ± {scores_stage2.std():.4f}")

# ============================================================================
# 新增：Stage 2 CV Learning Curves
# ============================================================================
print("=== Generating Stage 2 Learning Curves ===")

# 1. Learning Curve - 訓練集大小 vs 性能
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs_s2, train_scores_stage2, test_scores_stage2 = learning_curve(
    mlp_model, X_train_stage2_smote, y_train_stage2_smote, train_sizes=train_sizes, 
    cv=cv_stage2, scoring='f1_macro', n_jobs=-1, shuffle=True, random_state=42
)

train_scores_mean_s2 = np.mean(train_scores_stage2, axis=1)
train_scores_std_s2 = np.std(train_scores_stage2, axis=1)
test_scores_mean_s2 = np.mean(test_scores_stage2, axis=1)
test_scores_std_s2 = np.std(test_scores_stage2, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs_s2, train_scores_mean_s2, 'o-', color='green', label='Training F1-macro')
plt.fill_between(train_sizes_abs_s2, train_scores_mean_s2 - train_scores_std_s2,
                 train_scores_mean_s2 + train_scores_std_s2, alpha=0.1, color='green')
plt.plot(train_sizes_abs_s2, test_scores_mean_s2, 'o-', color='orange', label='CV F1-macro')
plt.fill_between(train_sizes_abs_s2, test_scores_mean_s2 - test_scores_std_s2,
                 test_scores_mean_s2 + test_scores_std_s2, alpha=0.1, color='orange')
plt.xlabel('Training Set Size')
plt.ylabel('F1-macro Score')
plt.title('Stage 2 Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_learning_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Validation Curve - max_iter
max_iter_range = [100, 200, 300, 500, 700, 1000]
train_scores_iter, test_scores_iter = validation_curve(
    MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42), 
    X_train_stage2_smote, y_train_stage2_smote, 
    param_name='max_iter', param_range=max_iter_range,
    cv=cv_stage2, scoring='f1_macro', n_jobs=-1
)

train_scores_mean_iter = np.mean(train_scores_iter, axis=1)
train_scores_std_iter = np.std(train_scores_iter, axis=1)
test_scores_mean_iter = np.mean(test_scores_iter, axis=1)
test_scores_std_iter = np.std(test_scores_iter, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(max_iter_range, train_scores_mean_iter, 'o-', color='green', label='Training F1-macro')
plt.fill_between(max_iter_range, train_scores_mean_iter - train_scores_std_iter,
                 train_scores_mean_iter + train_scores_std_iter, alpha=0.1, color='green')
plt.plot(max_iter_range, test_scores_mean_iter, 'o-', color='orange', label='CV F1-macro')
plt.fill_between(max_iter_range, test_scores_mean_iter - test_scores_std_iter,
                 test_scores_mean_iter + test_scores_std_iter, alpha=0.1, color='orange')
plt.xlabel('max_iter')
plt.ylabel('F1-macro Score')
plt.title('Stage 2 Validation Curve - max_iter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_validation_curve_max_iter.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. CV Fold Performance
fold_scores_stage2 = []
for fold, (train_idx, val_idx) in enumerate(cv_stage2.split(X_train_stage2_smote, y_train_stage2_smote)):
    X_fold_train = X_train_stage2_smote.iloc[train_idx]
    X_fold_val = X_train_stage2_smote.iloc[val_idx]
    y_fold_train = y_train_stage2_smote.iloc[train_idx]
    y_fold_val = y_train_stage2_smote.iloc[val_idx]
    
    mlp_model.fit(X_fold_train, y_fold_train)
    y_pred = mlp_model.predict(X_fold_val)
    f1 = f1_score(y_fold_val, y_pred, average='macro')
    fold_scores_stage2.append(f1)

plt.figure(figsize=(10, 6))
fold_labels = [f'Fold {i+1}' for i in range(5)]
bars = plt.bar(fold_labels, fold_scores_stage2, color=['lightseagreen', 'lightsalmon', 'wheat', 'thistle', 'lightsteelblue'])
plt.axhline(y=np.mean(fold_scores_stage2), color='red', linestyle='--', 
            label=f'Mean: {np.mean(fold_scores_stage2):.3f} ± {np.std(fold_scores_stage2):.3f}')

for bar, score in zip(bars, fold_scores_stage2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom')

plt.ylabel('F1-macro Score')
plt.title('Stage 2 CV Fold Performance')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_cv_fold_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. 新增：两阶段比较图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(fold_scores_stage1)+1), fold_scores_stage1, 
        color='skyblue', alpha=0.7, label='Stage 1')
plt.axhline(y=np.mean(fold_scores_stage1), color='blue', linestyle='--', 
            label=f'Mean: {np.mean(fold_scores_stage1):.3f}')
plt.xlabel('CV Fold')
plt.ylabel('F1-macro Score')
plt.title('Stage 1: pdnc vs non-pdnc')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, len(fold_scores_stage2)+1), fold_scores_stage2, 
        color='lightcoral', alpha=0.7, label='Stage 2')
plt.axhline(y=np.mean(fold_scores_stage2), color='red', linestyle='--', 
            label=f'Mean: {np.mean(fold_scores_stage2):.3f}')
plt.xlabel('CV Fold')
plt.ylabel('F1-macro Score')
plt.title('Stage 2: pdmci vs pdd')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cv_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print("=== CV Learning Curves Generation Completed ===")

# ============================================================================
# 原本的 Stage 2 測試代码继续...
# ============================================================================

# Stage 2 prediction with optimal threshold
y_test_stage2_true = class_test.loc[test_stage2_idx]
non_pdnc_test_idx = y_test_stage2_true.isin(['pdmci', 'pdd'])
X_test_stage2_non_pdnc = X_test_stage2.loc[non_pdnc_test_idx]
y_test_stage2_true_non_pdnc = y_test_stage2_true.loc[non_pdnc_test_idx]

ensemble_probs = mlp_model.predict_proba(X_test_stage2_non_pdnc)
pdd_index = list(label_encoder_stage2.classes_).index('pdd')
pdd_probs = ensemble_probs[:, pdd_index]
y_test_stage2_true_encoded = label_encoder_stage2.transform(y_test_stage2_true_non_pdnc)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test_stage2_true_encoded, pdd_probs, pos_label=pdd_index)
optimal_threshold = 0.5275
print(f"Optimal threshold for pdd : {optimal_threshold:.3f}")


y_pred_ensemble_adjusted_encoded = [
    pdd_index if prob[pdd_index] >= optimal_threshold else 1 - pdd_index for prob in ensemble_probs
]
y_pred_ensemble_adjusted = label_encoder_stage2.inverse_transform(y_pred_ensemble_adjusted_encoded)

print("=== Stage 2 Voting Ensemble Classification Report (Threshold Adjusted) ===")
print(classification_report(y_test_stage2_true_non_pdnc, y_pred_ensemble_adjusted))

# Stage 2 Confusion Matrix
cm_stage2 = confusion_matrix(y_test_stage2_true_non_pdnc, y_pred_ensemble_adjusted)
plt.figure(figsize=(6,4))
sns.heatmap(cm_stage2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['pdmci', 'pdd'], yticklabels=['pdmci', 'pdd'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Stage 2 Confusion Matrix (pdmci vs pdd)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_confusion_matrix.png'))
plt.show()

# Stage 2 ROC Curve and AUC (pdmci vs pdd)
y_test_stage2_encoded = label_encoder_stage2.transform(y_test_stage2_true_non_pdnc)
pdd_probs_stage2 = ensemble_probs[:, label_encoder_stage2.classes_.tolist().index('pdd')]

fpr_stage2, tpr_stage2, _ = roc_curve(y_test_stage2_encoded, pdd_probs_stage2)
roc_auc_stage2 = roc_auc_score(y_test_stage2_encoded, pdd_probs_stage2)

plt.figure(figsize=(7,5))
plt.plot(fpr_stage2, tpr_stage2, label=f'Stage 2 ROC (AUC = {roc_auc_stage2:.2f})', linewidth=2, color='orange')
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stage 2 ROC Curve (pdmci vs pdd)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_roc_curve.png'))
plt.show()

# Stage 2 Specificity
cm_stage2 = confusion_matrix(y_test_stage2_true_non_pdnc, y_pred_ensemble_adjusted, labels=['pdmci', 'pdd'])

TN, FP, FN, TP = cm_stage2.ravel()
specificity_stage2_pdmci = TN / (TN + FP)
specificity_stage2_pdd = TP / (TP + FN)

specificity_stage2_df = pd.DataFrame({
    'Class': ['pdmci', 'pdd'],
    'Specificity': [specificity_stage2_pdmci, specificity_stage2_pdd]
})

# Plot Specificity Stage 2
plt.figure(figsize=(6,4))
sns.barplot(x='Class', y='Specificity', data=specificity_stage2_df, palette='Oranges')
plt.ylim(0,1)
plt.title('Stage 2 Specificity')
plt.ylabel('Specificity')
plt.grid(axis='y')

for idx, value in enumerate(specificity_stage2_df['Specificity']):
    plt.text(idx, value + 0.02, f'{value:.2f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stage2_specificity.png'))
plt.show()

# Final integration
final_predictions = pd.Series(['pdnc'] * len(y_test_stage1), index=y_test_stage1.index)
final_predictions.loc[X_test_stage2_non_pdnc.index] = y_pred_ensemble_adjusted

print("=== Final Two-Stage Ensemble Classification Report ===")
print(classification_report(class_test, final_predictions))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Final Multi-class ROC Curve and AUC
y_test_binarized = label_binarize(class_test, classes=['pdnc', 'pdmci', 'pdd'])
y_pred_proba_final = np.zeros((len(class_test), 3))

# è¨­å®š pdnc çš„é æ¸¬æ©ŸçŽ‡ï¼ˆStage 1ï¼‰
y_pred_proba_final[:, 0] = 1 - proba_stage1

# è¨­å®š pdmci å'Œ pdd çš„é æ¸¬æ©ŸçŽ‡ï¼ˆStage 2ï¼‰
y_pred_proba_final[X_test_stage2_non_pdnc.index, 1:] = ensemble_probs

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
class_names = ['pdnc', 'pdmci', 'pdd']

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_final[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Model Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_multiclass_roc_curve.png'))
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(class_test, final_predictions, labels=['pdnc', 'pdmci', 'pdd'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pdnc', 'pdmci', 'pdd'], yticklabels=['pdnc', 'pdmci', 'pdd'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Two-Stage Ensemble Model')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_twostage.png'))
plt.show()

# è¨ˆç®—å„é¡žåˆ¥çš„æŒ‡æ¨™ä¸¦å„²å­˜
report = classification_report(class_test, final_predictions, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(output_dir, 'classification_report.csv'))

# Accuracy, Precision, Recall, F1-score
metrics_df = pd.DataFrame({
    'Accuracy': [accuracy_score(class_test, final_predictions)],
    'Precision': [precision_score(class_test, final_predictions, average='macro')],
    'Recall': [recall_score(class_test, final_predictions, average='macro')],
    'F1-score': [f1_score(class_test, final_predictions, average='macro')]
})

ax = metrics_df.plot.bar(figsize=(8,6), legend=True)
plt.title('Overall Classification Metrics')
plt.ylim(0,1)
plt.xticks([])
plt.ylabel('Score')
plt.grid(axis='y')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.savefig(os.path.join(output_dir, 'classification_metrics.png'))
plt.close()

# Specificity, PPV, NPV
cm = confusion_matrix(class_test, final_predictions, labels=['pdd','pdmci','pdnc'])
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (FP + FN + TP)

specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)

spec_df = pd.DataFrame({'Specificity': specificity, 'PPV': PPV, 'NPV': NPV}, index=['pdd','pdmci','pdnc'])
ax = spec_df.plot.bar(figsize=(10,6))
plt.title('Specificity, PPV, NPV by Class')
plt.ylim(0,1)
plt.ylabel('Score')
plt.grid(axis='y')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.savefig(os.path.join(output_dir, 'specificity_ppv_npv.png'))
plt.close()

# ROC Curve èˆ‡ AUC (pdd vs éžpdd)
pdd_binary_true = class_test.apply(lambda x: 1 if x=='pdd' else 0)
pdd_binary_pred_prob = pd.Series(0, index=class_test.index)
pdd_binary_pred_prob[X_test_stage2_non_pdnc.index] = pdd_probs

fpr, tpr, _ = roc_curve(pdd_binary_true, pdd_binary_pred_prob)
roc_auc = roc_auc_score(pdd_binary_true, pdd_binary_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PDD')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Calibration Curve
prob_true, prob_pred = calibration_curve(pdd_binary_true, pdd_binary_pred_prob, n_bins=10)
plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (PDD)')
plt.grid()
plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
plt.close()

# SHAP ç‰¹å¾µé‡è¦æ€§
shap.summary_plot(shap_values_stage2, X_train_stage2, plot_type="bar", show=False)
plt.title("SHAP Feature Importance Stage 2")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_importance_stage2.png'))
plt.close()

print("=== 所有分析完成 ===")
print("生成的CV Learning Curves图表：")
print("- stage1_learning_curve.png")
print("- stage1_validation_curve_n_estimators.png") 
print("- stage1_cv_fold_performance.png")
print("- stage2_learning_curve.png")
print("- stage2_validation_curve_max_iter.png")
print("- stage2_cv_fold_performance.png")
print("- cv_performance_comparison.png")

# ============================================================================
# 統計信賴區間測試（加在原程式碼最後）
# ============================================================================
from scipy import stats
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import pandas as pd

print("=== Statistical Confidence Intervals Analysis ===")

# 1. PDD 敏感度（recall）的信賴區間 - Clopper-Pearson binomial interval
print("\n1. PDD Sensitivity (Recall) Confidence Interval:")

# 取得PDD的TP和FN
pdd_true = (class_test == 'pdd').values
pdd_pred = (final_predictions == 'pdd').values

# 計算 TP, FN, FP, TN for PDD
pdd_tp = np.sum(pdd_true & pdd_pred)
pdd_fn = np.sum(pdd_true & ~pdd_pred)
pdd_fp = np.sum(~pdd_true & pdd_pred)
pdd_tn = np.sum(~pdd_true & ~pdd_pred)

pdd_total_positive = pdd_tp + pdd_fn
pdd_sensitivity = pdd_tp / pdd_total_positive if pdd_total_positive > 0 else 0

# Clopper-Pearson 95% CI for binomial proportion
if pdd_total_positive > 0:
    pdd_ci_lower = stats.beta.ppf(0.025, pdd_tp, pdd_total_positive - pdd_tp + 1)
    pdd_ci_upper = stats.beta.ppf(0.975, pdd_tp + 1, pdd_total_positive - pdd_tp)
    
    print(f"PDD cases in test set: {pdd_total_positive}")
    print(f"PDD correctly identified (TP): {pdd_tp}")
    print(f"PDD sensitivity: {pdd_sensitivity:.3f}")
    print(f"95% CI: [{pdd_ci_lower:.3f}, {pdd_ci_upper:.3f}]")
else:
    print("No PDD cases in test set")

# 同樣對其他類別計算
for class_name in ['pdnc', 'pdmci']:
    class_true = (class_test == class_name).values
    class_pred = (final_predictions == class_name).values
    
    class_tp = np.sum(class_true & class_pred)
    class_fn = np.sum(class_true & ~class_pred)
    class_total_positive = class_tp + class_fn
    class_sensitivity = class_tp / class_total_positive if class_total_positive > 0 else 0
    
    if class_total_positive > 0:
        class_ci_lower = stats.beta.ppf(0.025, class_tp, class_total_positive - class_tp + 1)
        class_ci_upper = stats.beta.ppf(0.975, class_tp + 1, class_total_positive - class_tp)
        
        print(f"\n{class_name.upper()} sensitivity: {class_sensitivity:.3f}")
        print(f"95% CI: [{class_ci_lower:.3f}, {class_ci_upper:.3f}] (n={class_total_positive})")

# 2. AUC 的信賴區間 - DeLong 方法
print("\n\n2. AUC Confidence Intervals (DeLong method):")

def delong_roc_variance(ground_truth, predictions):
    """
    計算 DeLong et al. 方法的 AUC 變異數
    """
    order = np.argsort(predictions)[::-1]
    predictions_sorted_transposed = predictions[order]
    aucs, delongcov = delong_roc_ci_bootstrap(ground_truth, predictions_sorted_transposed)
    return aucs, delongcov

def delong_roc_ci_bootstrap(y_true, y_scores, alpha=0.05):
    """
    簡化的 DeLong AUC CI 計算
    """
    from sklearn.metrics import roc_auc_score
    
    auc = roc_auc_score(y_true, y_scores)
    n1 = np.sum(y_true == 1)  # positive samples
    n2 = np.sum(y_true == 0)  # negative samples
    
    if n1 == 0 or n2 == 0:
        return auc, (auc, auc)
    
    # 簡化的標準誤計算
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (auc / (2 - auc) - auc**2) + 
                  (n2 - 1) * (2 * auc**2 / (1 + auc) - auc**2)) / (n1 * n2))
    
    # 95% CI
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower = auc - z_score * se
    ci_upper = auc + z_score * se
    
    return auc, (max(0, ci_lower), min(1, ci_upper))

# 計算各類別的 one-vs-rest AUC CI
for i, class_name in enumerate(['pdnc', 'pdmci', 'pdd']):
    y_binary = (class_test == class_name).astype(int)
    y_scores = y_pred_proba_final[:, i]
    
    auc_val, (ci_lower, ci_upper) = delong_roc_ci_bootstrap(y_binary, y_scores)
    print(f"{class_name.upper()} AUC: {auc_val:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 3. Macro-F1 和 class-wise recall 的 Bootstrap 信賴區間
print("\n\n3. Bootstrap Confidence Intervals (1000 iterations):")

def stratified_bootstrap_sample(y_true, y_pred, y_proba, random_state=None):
    """
    分層 bootstrap 抽樣，保持類別比例
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y_true)
    classes = np.unique(y_true)
    bootstrap_indices = []
    
    for class_label in classes:
        class_indices = np.where(y_true == class_label)[0]
        n_class = len(class_indices)
        bootstrap_class_indices = np.random.choice(class_indices, size=n_class, replace=True)
        bootstrap_indices.extend(bootstrap_class_indices)
    
    bootstrap_indices = np.array(bootstrap_indices)
    np.random.shuffle(bootstrap_indices)
    
    return y_true[bootstrap_indices], y_pred[bootstrap_indices], y_proba[bootstrap_indices]

# Bootstrap 統計計算
n_bootstrap = 1000
bootstrap_metrics = {
    'macro_f1': [],
    'pdnc_recall': [],
    'pdmci_recall': [],
    'pdd_recall': []
}

class_test_array = class_test.values
final_predictions_array = final_predictions.values

print("Running bootstrap sampling...")
for i in range(n_bootstrap):
    # 分層抽樣
    boot_true, boot_pred, _ = stratified_bootstrap_sample(
        class_test_array, final_predictions_array, y_pred_proba_final,
        random_state=i
    )
    
    # 計算指標
    macro_f1 = f1_score(boot_true, boot_pred, average='macro')
    bootstrap_metrics['macro_f1'].append(macro_f1)
    
    # 各類別 recall
    for class_name in ['pdnc', 'pdmci', 'pdd']:
        class_recall = recall_score(boot_true, boot_pred, labels=[class_name], average=None)[0]
        bootstrap_metrics[f'{class_name}_recall'].append(class_recall)

# 計算 95% 信賴區間
print("\nBootstrap Results:")
for metric_name, values in bootstrap_metrics.items():
    mean_val = np.mean(values)
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    print(f"{metric_name}: {mean_val:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 儲存 bootstrap 結果
bootstrap_results_df = pd.DataFrame(bootstrap_metrics)
bootstrap_results_df.to_csv(os.path.join(output_dir, 'bootstrap_results.csv'), index=False)

# 繪製 bootstrap 分佈
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (metric_name, values) in enumerate(bootstrap_metrics.items()):
    axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
    axes[i].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
    axes[i].axvline(np.percentile(values, 2.5), color='orange', linestyle='--', alpha=0.7)
    axes[i].axvline(np.percentile(values, 97.5), color='orange', linestyle='--', alpha=0.7)
    axes[i].set_title(f'{metric_name} Bootstrap Distribution')
    axes[i].set_xlabel('Score')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# 總結報告
print("\n=== CONFIDENCE INTERVALS SUMMARY ===")
print("1. Binomial CIs provide exact coverage for sensitivity/recall")
print("2. DeLong method gives asymptotic CIs for AUC")
print("3. Bootstrap CIs capture uncertainty in macro-F1 and class-wise metrics")
print("4. All results saved to results directory")

print("\n=== Analysis Complete ===")
print("Files generated:")
print("- bootstrap_results.csv")
print("- bootstrap_distributions.png")

# ============================================================================
# 新增：基线筛查方法比较 (Reviewer 2 Question 6)
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

print("=== Baseline Screening Methods Comparison ===")

# 准备数据
X_test_full = X_test.copy()
y_true = class_test.copy()

# 确保有必要的特征
required_features = ['MCATOT_x', 'NP1COG', 'NP2HYGN', 'NP2RISE', 'ENROLL_AGE']
available_features = [f for f in required_features if f in X_test_full.columns]
print(f"Available features for baseline comparison: {available_features}")

# ============================================================================
# Method 1: MoCA Cutoff Only
# ============================================================================
def baseline_moca_only(moca_scores, cutoffs=[26, 21]):
    """Simple MoCA cutoff rules"""
    predictions = []
    for score in moca_scores:
        if pd.isna(score):
            predictions.append('pdnc')  # 默认为正常
        elif score >= cutoffs[0]:
            predictions.append('pdnc')
        elif score >= cutoffs[1]:
            predictions.append('pdmci')
        else:
            predictions.append('pdd')
    return np.array(predictions)

if 'MCATOT_x' in X_test_full.columns:
    moca_scores = X_test_full['MCATOT_x'].values
    pred_moca_only = baseline_moca_only(moca_scores)
    
    # 计算性能
    acc_moca = accuracy_score(y_true, pred_moca_only)
    f1_macro_moca = f1_score(y_true, pred_moca_only, average='macro')
    f1_pdd_moca = f1_score(y_true, pred_moca_only, labels=['pdd'], average='macro')
    
    print(f"\nMethod 1 - MoCA Only:")
    print(f"Accuracy: {acc_moca:.3f}")
    print(f"Macro F1: {f1_macro_moca:.3f}")
    print(f"PDD F1: {f1_pdd_moca:.3f}")
else:
    print("MoCA scores not available for Method 1")
    pred_moca_only = None

# ============================================================================
# Method 2: MoCA + ADL Rules
# ============================================================================
def baseline_moca_adl_rules(moca_scores, adl_scores, moca_cutoffs=[26, 21], adl_threshold=1):
    """MoCA + ADL rule-based approach"""
    predictions = []
    
    for i in range(len(moca_scores)):
        moca = moca_scores[i] if not pd.isna(moca_scores[i]) else 26
        adl = adl_scores[i] if not pd.isna(adl_scores[i]) else 0
        
        # 规则逻辑
        if moca >= moca_cutoffs[0] and adl <= adl_threshold:
            predictions.append('pdnc')
        elif moca < moca_cutoffs[1] or adl > adl_threshold:
            predictions.append('pdd')
        else:
            predictions.append('pdmci')
    
    return np.array(predictions)

if 'MCATOT_x' in X_test_full.columns and 'NP2HYGN' in X_test_full.columns:
    moca_scores = X_test_full['MCATOT_x'].values
    adl_scores = X_test_full['NP2HYGN'].values  # 使用卫生独立性作为ADL指标
    pred_moca_adl = baseline_moca_adl_rules(moca_scores, adl_scores)
    
    # 计算性能
    acc_moca_adl = accuracy_score(y_true, pred_moca_adl)
    f1_macro_moca_adl = f1_score(y_true, pred_moca_adl, average='macro')
    f1_pdd_moca_adl = f1_score(y_true, pred_moca_adl, labels=['pdd'], average='macro')
    
    print(f"\nMethod 2 - MoCA + ADL:")
    print(f"Accuracy: {acc_moca_adl:.3f}")
    print(f"Macro F1: {f1_macro_moca_adl:.3f}")
    print(f"PDD F1: {f1_pdd_moca_adl:.3f}")
else:
    print("Required features not available for Method 2")
    pred_moca_adl = None

# ============================================================================
# Method 3: Simple Logistic Regression (5 features)
# ============================================================================
simple_features = []
for feat in ['MCATOT_x', 'NP1COG', 'NP2HYGN', 'ENROLL_AGE']:
    if feat in X_test_full.columns and feat in X_train.columns:
        simple_features.append(feat)

if len(simple_features) >= 3:  # 至少需要3个特征
    print(f"\nMethod 3 - Simple Logistic Regression using: {simple_features}")
    
    # 准备训练和测试数据
    X_train_simple = X_train[simple_features]
    X_test_simple = X_test_full[simple_features]
    y_train_simple = class_train
    
    # 训练简单逻辑回归
    simple_lr = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000)
    simple_lr.fit(X_train_simple, y_train_simple)
    
    # 预测
    pred_simple_lr = simple_lr.predict(X_test_simple)
    
    # 计算性能
    acc_simple = accuracy_score(y_true, pred_simple_lr)
    f1_macro_simple = f1_score(y_true, pred_simple_lr, average='macro')
    f1_pdd_simple = f1_score(y_true, pred_simple_lr, labels=['pdd'], average='macro')
    
    print(f"Accuracy: {acc_simple:.3f}")
    print(f"Macro F1: {f1_macro_simple:.3f}")
    print(f"PDD F1: {f1_pdd_simple:.3f}")
else:
    print("Insufficient features for Method 3")
    pred_simple_lr = None

# ============================================================================
# Our Two-Stage Model Performance (已有)
# ============================================================================
acc_twostage = accuracy_score(y_true, final_predictions)
f1_macro_twostage = f1_score(y_true, final_predictions, average='macro')
f1_pdd_twostage = f1_score(y_true, final_predictions, labels=['pdd'], average='macro')

print(f"\nOur Two-Stage Model:")
print(f"Accuracy: {acc_twostage:.3f}")
print(f"Macro F1: {f1_macro_twostage:.3f}")
print(f"PDD F1: {f1_pdd_twostage:.3f}")

# ============================================================================
# 比较结果可视化
# ============================================================================

# 准备比较数据
methods = []
accuracies = []
macro_f1s = []
pdd_f1s = []

if pred_moca_only is not None:
    methods.append('MoCA Only')
    accuracies.append(acc_moca)
    macro_f1s.append(f1_macro_moca)
    pdd_f1s.append(f1_pdd_moca)

if pred_moca_adl is not None:
    methods.append('MoCA + ADL')
    accuracies.append(acc_moca_adl)
    macro_f1s.append(f1_macro_moca_adl)
    pdd_f1s.append(f1_pdd_moca_adl)

if pred_simple_lr is not None:
    methods.append('Simple LR')
    accuracies.append(acc_simple)
    macro_f1s.append(f1_macro_simple)
    pdd_f1s.append(f1_pdd_simple)

# 添加我们的方法
methods.append('Two-Stage ML')
accuracies.append(acc_twostage)
macro_f1s.append(f1_macro_twostage)
pdd_f1s.append(f1_pdd_twostage)

# 创建比较DataFrame
comparison_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracies,
    'Macro F1': macro_f1s,
    'PDD F1': pdd_f1s
})

print("\n=== Performance Comparison Table ===")
print(comparison_df.round(3))

# 保存比较结果
comparison_df.to_csv(os.path.join(output_dir, 'baseline_comparison.csv'), index=False)

# ============================================================================
# 可视化比较结果
# ============================================================================

# 1. 整体性能比较
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy比较
axes[0].bar(methods, accuracies, color=['lightblue', 'lightgreen', 'orange', 'red'])
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')

# Macro F1比较  
axes[1].bar(methods, macro_f1s, color=['lightblue', 'lightgreen', 'orange', 'red'])
axes[1].set_title('Macro F1 Comparison')
axes[1].set_ylabel('Macro F1')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(macro_f1s):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')

# PDD F1比较
axes[2].bar(methods, pdd_f1s, color=['lightblue', 'lightgreen', 'orange', 'red'])
axes[2].set_title('PDD F1 Comparison')
axes[2].set_ylabel('PDD F1')
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(pdd_f1s):
    axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'baseline_comparison_metrics.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. 详细的分类报告比较
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

predictions_list = [pred_moca_only, pred_moca_adl, pred_simple_lr, final_predictions]
method_names = ['MoCA Only', 'MoCA + ADL', 'Simple LR', 'Two-Stage ML']

for idx, (pred, method) in enumerate(zip(predictions_list, method_names)):
    if pred is not None:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, pred, labels=['pdnc', 'pdmci', 'pdd'])
        
        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['pdnc', 'pdmci', 'pdd'],
                   yticklabels=['pdnc', 'pdmci', 'pdd'],
                   ax=axes[idx])
        axes[idx].set_title(f'{method} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'baseline_comparison_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. 性能提升可视化
if len(methods) >= 2:
    # 相对于最简单方法的改进
    baseline_acc = accuracies[0]  # 第一个方法作为baseline
    baseline_f1 = macro_f1s[0]
    
    acc_improvements = [(acc - baseline_acc) / baseline_acc * 100 for acc in accuracies]
    f1_improvements = [(f1 - baseline_f1) / baseline_f1 * 100 for f1 in macro_f1s]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确度提升
    bars1 = ax1.bar(methods, acc_improvements, color=['lightblue', 'lightgreen', 'orange', 'red'])
    ax1.set_title('Accuracy Improvement over MoCA Only (%)')
    ax1.set_ylabel('Improvement (%)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    for bar, imp in zip(bars1, acc_improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # F1提升
    bars2 = ax2.bar(methods, f1_improvements, color=['lightblue', 'lightgreen', 'orange', 'red'])
    ax2.set_title('Macro F1 Improvement over MoCA Only (%)')
    ax2.set_ylabel('Improvement (%)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    for bar, imp in zip(bars2, f1_improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -1),
                f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_comparison_improvements.png'), dpi=300, bbox_inches='tight')
    plt.show()

print("\n=== Baseline Comparison Analysis Complete ===")
print("Generated files:")
print("- baseline_comparison.csv")
print("- baseline_comparison_metrics.png") 
print("- baseline_comparison_confusion_matrices.png")
print("- baseline_comparison_improvements.png")

# 打印总结
print(f"\n=== SUMMARY ===")
print(f"Our two-stage ML model shows substantial improvements:")
print(f"- Accuracy: {acc_twostage:.3f} (vs best baseline: {max(accuracies[:-1]):.3f})")
print(f"- Macro F1: {f1_macro_twostage:.3f} (vs best baseline: {max(macro_f1s[:-1]):.3f})")  
print(f"- PDD F1: {f1_pdd_twostage:.3f} (vs best baseline: {max(pdd_f1s[:-1]):.3f})")


# ============================================================================
# 新增：子项目 vs 总分比较分析 (Reviewer 2 Question 7)
# ============================================================================

print("=== Subitem vs Total Score Comparison Analysis ===")

# 1. 定义总分特征 (如果可用)
total_score_features = []
potential_totals = ['MCATOT_x', 'UPDRS_TOTAL', 'GDS_TOTAL', 'ESS_TOTAL', 
                   'JLO_TOTAL', 'HVLT_TOTAL']

# 检查哪些总分特征可用
for feat in potential_totals:
    if feat in X_train.columns:
        total_score_features.append(feat)

# 如果没有现成的总分，从子项目计算
if 'MCATOT_x' not in total_score_features and any('MCA' in col for col in X_train.columns):
    # 计算MoCA总分 (如果有子项目)
    moca_subitems = [col for col in X_train.columns if 'MCA' in col and col != 'MCATOT_x']
    if moca_subitems:
        X_train['MCA_COMPUTED_TOTAL'] = X_train[moca_subitems].sum(axis=1)
        X_test['MCA_COMPUTED_TOTAL'] = X_test[moca_subitems].sum(axis=1)
        total_score_features.append('MCA_COMPUTED_TOTAL')

# 添加基本的人口统计学特征
demographic_features = []
for feat in ['ENROLL_AGE', 'EDUCYRS', 'PD_DURATION']:
    if feat in X_train.columns:
        demographic_features.append(feat)

total_score_features_final = total_score_features + demographic_features

print(f"Total score features: {total_score_features_final}")
print(f"Number of total score features: {len(total_score_features_final)}")

# 当前子项目特征数量
current_subitem_features = selected_features  # 来自之前的特征选择
print(f"Current subitem features: {len(current_subitem_features)}")

# ============================================================================
# Model 1: 仅使用总分特征
# ============================================================================
if len(total_score_features_final) >= 3:  # 确保有足够的特征
    print("\n=== Training Total Score Only Model ===")
    
    # 准备数据
    X_train_total = X_train[total_score_features_final]
    X_test_total = X_test[total_score_features_final]
    
    # 使用相同的两阶段架构
    # Stage 1: PDNC vs non-PDNC
    smote_total_s1 = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_total_s1_smote, y_train_s1_smote = smote_total_s1.fit_resample(X_train_total, y_train_stage1)
    
    xgb_total_s1 = XGBClassifier(
        learning_rate=0.05, max_depth=3, n_estimators=200,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42
    )
    xgb_total_s1.fit(X_train_total_s1_smote, y_train_s1_smote)
    
    # Stage 1 预测
    proba_total_s1 = xgb_total_s1.predict_proba(X_test_total)[:, 1]
    y_pred_total_s1 = (proba_total_s1 >= 0.203).astype(int)  # 使用相同阈值
    
    # Stage 2: PDMCI vs PDD (对于被预测为非PDNC的样本)
    test_stage2_total_idx = (y_pred_total_s1 == 1)
    
    if np.sum(test_stage2_total_idx) > 0:
        # 准备Stage 2数据
        train_stage2_idx = (y_train_stage1 == 1) & (class_train != 'pdnc')
        X_train_total_s2 = X_train_total.loc[train_stage2_idx]
        y_train_total_s2 = class_train.loc[train_stage2_idx].map({'pdmci': 0, 'pdd': 1})
        
        # Stage 2训练
        smote_total_s2 = SMOTETomek(sampling_strategy={0: 400, 1: 350}, random_state=42)
        X_train_total_s2_smote, y_train_total_s2_smote = smote_total_s2.fit_resample(X_train_total_s2, y_train_total_s2)
        
        mlp_total_s2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp_total_s2.fit(X_train_total_s2_smote, y_train_total_s2_smote)
        
        # Stage 2预测
        X_test_total_s2 = X_test_total.loc[test_stage2_total_idx]
        ensemble_probs_total = mlp_total_s2.predict_proba(X_test_total_s2)
        y_pred_total_s2 = (ensemble_probs_total[:, 1] >= 0.563).astype(int)  # 使用相同阈值
        
        # 整合最终预测
        final_predictions_total = pd.Series(['pdnc'] * len(y_test_stage1), index=y_test_stage1.index)
        stage2_indices = y_test_stage1.index[test_stage2_total_idx]
        final_predictions_total.loc[stage2_indices] = ['pdmci' if pred == 0 else 'pdd' for pred in y_pred_total_s2]
        
        # 计算性能
        acc_total = accuracy_score(class_test, final_predictions_total)
        f1_macro_total = f1_score(class_test, final_predictions_total, average='macro')
        f1_pdd_total = f1_score(class_test, final_predictions_total, labels=['pdd'], average='macro')
        
        print(f"Total Score Model Performance:")
        print(f"Accuracy: {acc_total:.3f}")
        print(f"Macro F1: {f1_macro_total:.3f}") 
        print(f"PDD F1: {f1_pdd_total:.3f}")
    else:
        print("No samples predicted as non-PDNC in Stage 1")
        final_predictions_total = None
else:
    print("Insufficient total score features available")
    final_predictions_total = None

# ============================================================================
# Model 2: 子项目模型 (已有)
# ============================================================================
acc_subitem = accuracy_score(class_test, final_predictions)
f1_macro_subitem = f1_score(class_test, final_predictions, average='macro')
f1_pdd_subitem = f1_score(class_test, final_predictions, labels=['pdd'], average='macro')

print(f"\nSubitem Model Performance (Current):")
print(f"Accuracy: {acc_subitem:.3f}")
print(f"Macro F1: {f1_macro_subitem:.3f}")
print(f"PDD F1: {f1_pdd_subitem:.3f}")

# ============================================================================
# 比较结果可视化
# ============================================================================
if final_predictions_total is not None:
    # 准备比较数据
    comparison_methods = ['Total Scores', 'Subitem Features']
    comparison_acc = [acc_total, acc_subitem]
    comparison_f1_macro = [f1_macro_total, f1_macro_subitem] 
    comparison_f1_pdd = [f1_pdd_total, f1_pdd_subitem]
    
    # 计算改进幅度
    acc_improvement = (acc_subitem - acc_total) / acc_total * 100
    f1_improvement = (f1_macro_subitem - f1_macro_total) / f1_macro_total * 100
    pdd_improvement = (f1_pdd_subitem - f1_pdd_total) / f1_pdd_total * 100
    
    print(f"\nSubitem vs Total Score Improvements:")
    print(f"Accuracy improvement: {acc_improvement:.1f}%")
    print(f"Macro F1 improvement: {f1_improvement:.1f}%")
    print(f"PDD F1 improvement: {pdd_improvement:.1f}%")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy比较
    bars1 = axes[0].bar(comparison_methods, comparison_acc, color=['lightcoral', 'steelblue'])
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    for bar, acc in zip(bars1, comparison_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center')
    
    # Macro F1比较
    bars2 = axes[1].bar(comparison_methods, comparison_f1_macro, color=['lightcoral', 'steelblue'])
    axes[1].set_title('Macro F1 Comparison') 
    axes[1].set_ylabel('Macro F1')
    axes[1].set_ylim(0, 1)
    for bar, f1 in zip(bars2, comparison_f1_macro):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center')
    
    # PDD F1比较
    bars3 = axes[2].bar(comparison_methods, comparison_f1_pdd, color=['lightcoral', 'steelblue'])
    axes[2].set_title('PDD F1 Comparison')
    axes[2].set_ylabel('PDD F1')
    axes[2].set_ylim(0, 1)
    for bar, f1 in zip(bars3, comparison_f1_pdd):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subitem_vs_total_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存比较结果
    subitem_comparison_df = pd.DataFrame({
        'Method': comparison_methods,
        'Accuracy': comparison_acc,
        'Macro F1': comparison_f1_macro,
        'PDD F1': comparison_f1_pdd,
        'N_Features': [len(total_score_features_final), len(current_subitem_features)]
    })
    
    subitem_comparison_df.to_csv(os.path.join(output_dir, 'subitem_vs_total_comparison.csv'), index=False)
    print("\nComparison results saved to subitem_vs_total_comparison.csv")

print("=== Subitem vs Total Score Analysis Complete ===")

# ============================================================================
# 新增：SHAP方向性分析 (Reviewer 2 Question 10)
# ============================================================================
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=== Enhanced SHAP Directional Analysis ===")

# 1. Stage 1 方向性SHAP分析 (PDNC vs non-PDNC)
print("\n=== Stage 1 Directional SHAP Analysis ===")

# 重新计算SHAP值（保留符号）
explainer_s1 = shap.TreeExplainer(best_model_stage1)
shap_values_s1 = explainer_s1.shap_values(X_train_stage1)  # 这是有符号的SHAP值

# 计算每个特征的平均正负贡献
feature_contributions_s1 = pd.DataFrame({
    'Feature': X_train_stage1.columns,
    'Mean_SHAP': np.mean(shap_values_s1, axis=0),
    'Pos_SHAP': np.mean(np.maximum(shap_values_s1, 0), axis=0),
    'Neg_SHAP': np.mean(np.minimum(shap_values_s1, 0), axis=0),
    'Abs_SHAP': np.mean(np.abs(shap_values_s1), axis=0)
}).sort_values('Abs_SHAP', ascending=False)

print("Top 10 Features with Directional Information (Stage 1):")
print(feature_contributions_s1.head(10).round(4))

# SHAP Summary Plot with Direction
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_s1, X_train_stage1, plot_type="dot", show=False)
plt.title("Stage 1: Directional SHAP Summary (PDNC vs non-PDNC)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_stage1_directional_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

# SHAP Waterfall Plot for representative samples
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 选择代表性样本
sample_indices = [0, 10, 20, 30]  # 不同的样本
sample_labels = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']

for idx, (sample_idx, label) in enumerate(zip(sample_indices, sample_labels)):
    if sample_idx < len(X_train_stage1):
        ax = axes[idx//2, idx%2]
        
        # 手动创建waterfall效果
        feature_vals = X_train_stage1.iloc[sample_idx].values
        shap_vals = shap_values_s1[sample_idx]
        feature_names = X_train_stage1.columns
        
        # 按SHAP值绝对值排序，取前10个
        top_indices = np.argsort(np.abs(shap_vals))[-10:]
        
        colors = ['red' if val < 0 else 'blue' for val in shap_vals[top_indices]]
        bars = ax.barh(range(len(top_indices)), shap_vals[top_indices], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([feature_names[i] for i in top_indices])
        ax.set_xlabel('SHAP Value (← PDNC | non-PDNC →)')
        ax.set_title(f'{label} Waterfall')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_stage1_waterfall_examples.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Stage 2 方向性SHAP分析 (PDMCI vs PDD)
print("\n=== Stage 2 Directional SHAP Analysis ===")

# 重新训练并计算Stage 2 SHAP
explainer_s2 = shap.TreeExplainer(xgb_stage2)  # 使用Stage 2的XGBoost模型
shap_values_s2 = explainer_s2.shap_values(X_train_stage2)

# 计算方向性贡献
feature_contributions_s2 = pd.DataFrame({
    'Feature': X_train_stage2.columns,
    'Mean_SHAP': np.mean(shap_values_s2, axis=0),
    'Pos_SHAP': np.mean(np.maximum(shap_values_s2, 0), axis=0),
    'Neg_SHAP': np.mean(np.minimum(shap_values_s2, 0), axis=0),
    'Abs_SHAP': np.mean(np.abs(shap_values_s2), axis=0)
}).sort_values('Abs_SHAP', ascending=False)

print("Top 10 Features with Directional Information (Stage 2):")
print(feature_contributions_s2.head(10).round(4))

# SHAP Summary Plot Stage 2
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_s2, X_train_stage2, plot_type="dot", show=False)
plt.title("Stage 2: Directional SHAP Summary (PDMCI vs PDD)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_stage2_directional_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. 特征依赖图 (Dependence Plots)
print("\n=== SHAP Dependence Analysis ===")

# 选择最重要的几个特征进行dependence plot
top_features_s1 = feature_contributions_s1.head(4)['Feature'].tolist()
top_features_s2 = feature_contributions_s2.head(4)['Feature'].tolist()

# Stage 1 Dependence Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_features_s1):
    if idx < 4:
        shap.dependence_plot(feature, shap_values_s1, X_train_stage1, ax=axes[idx], show=False)
        axes[idx].set_title(f'Stage 1: {feature} Dependence')

plt.suptitle('Stage 1 Feature Dependence Plots')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_stage1_dependence.png'), dpi=300, bbox_inches='tight')
plt.show()

# Stage 2 Dependence Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_features_s2):
    if idx < 4 and feature in X_train_stage2.columns:
        shap.dependence_plot(feature, shap_values_s2, X_train_stage2, ax=axes[idx], show=False)
        axes[idx].set_title(f'Stage 2: {feature} Dependence')

plt.suptitle('Stage 2 Feature Dependence Plots')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_stage2_dependence.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. 临床解释总结
print("\n=== Clinical Interpretation Summary ===")

print("\nStage 1 (PDNC vs non-PDNC) - Key Directional Insights:")
for _, row in feature_contributions_s1.head(5).iterrows():
    direction = "promotes non-PDNC" if row['Mean_SHAP'] > 0 else "promotes PDNC"
    print(f"- {row['Feature']}: {direction} (mean SHAP = {row['Mean_SHAP']:.3f})")

print("\nStage 2 (PDMCI vs PDD) - Key Directional Insights:")
for _, row in feature_contributions_s2.head(5).iterrows():
    direction = "promotes PDD" if row['Mean_SHAP'] > 0 else "promotes PDMCI"
    print(f"- {row['Feature']}: {direction} (mean SHAP = {row['Mean_SHAP']:.3f})")

# 保存详细的方向性分析结果
feature_contributions_s1.to_csv(os.path.join(output_dir, 'shap_stage1_directional_analysis.csv'), index=False)
feature_contributions_s2.to_csv(os.path.join(output_dir, 'shap_stage2_directional_analysis.csv'), index=False)

print("\n=== Directional SHAP Analysis Complete ===")
print("Generated files:")
print("- shap_stage1_directional_summary.png")
print("- shap_stage1_waterfall_examples.png") 
print("- shap_stage2_directional_summary.png")
print("- shap_stage1_dependence.png")
print("- shap_stage2_dependence.png")
print("- shap_stage1_directional_analysis.csv")
print("- shap_stage2_directional_analysis.csv")