#%%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

base = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Base.csv')
base_copy = base.copy()

var1 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant I.csv')
var1_copy = var1.copy()

var2 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant II.csv')
var2_copy = var2.copy()

var3 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant III.csv')
var3_copy = var3.copy()

var4 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant IV.csv')
var4_copy = var4.copy()

var5 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant V.csv')
var5_copy = var5.copy()

#%%
### EDA dataset preprocessing ###
def EDA_dataset(df):
    drop_col = ['payment_type', 'employment_status', 'prev_address_months_count', 'intended_balcon_amount', 'housing_status', 'days_since_request']
    df.drop(columns = drop_col, inplace = True)

    df = df[df['current_address_months_count'] >= 0]

    df['bank_months_count'].replace({-1: 0}, inplace = True)

    df = df[df['session_length_in_minutes'] >= 0]

    df['proposed_credit_limit'] = df['proposed_credit_limit'].astype(int)


    return df
# %%
base_copy = EDA_dataset(base_copy)
var1_copy = EDA_dataset(var1_copy)
var2_copy = EDA_dataset(var2_copy)
var3_copy = EDA_dataset(var3_copy)
var4_copy = EDA_dataset(var4_copy)
var5_copy = EDA_dataset(var5_copy)

# %%
### One-hot encoding for categorical variables####
def one_hot(df):
    object_cols = ['source', 'device_os']
    df = pd.get_dummies(df, columns=object_cols, drop_first=True, dtype=int)

    return df
# %%
base_copy = one_hot(base_copy)
var1_copy = one_hot(var1_copy)  
var2_copy = one_hot(var2_copy)
var3_copy = one_hot(var3_copy)
var4_copy = one_hot(var4_copy)
var5_copy = one_hot(var5_copy)


#%%
###100만개 데이터 중 10만개씩 sampling###
from sklearn.model_selection import train_test_split
def sample_data_stratified(df):
    df.loc[(df['customer_age'] < 50) & (df['fraud_bool'] == 0), 'group'] = 0
    df.loc[(df['customer_age'] < 50) & (df['fraud_bool'] == 1), 'group'] = 1
    df.loc[(df['customer_age'] >= 50) & (df['fraud_bool'] == 0), 'group'] = 2
    df.loc[(df['customer_age'] >= 50) & (df['fraud_bool'] == 1), 'group'] = 3
    df['group'] = df['group'].astype(int)

    X = df.drop(columns=['group'])
    y = df['group']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, X_test, y_train, y_test
#%%
base_sam, X_test, y_train, y_test = sample_data_stratified(base_copy)
var1_sam, X_test, y_train, y_test = sample_data_stratified(var1_copy)
var2_sam, X_test, y_train, y_test = sample_data_stratified(var2_copy)
var3_sam, X_test, y_train, y_test = sample_data_stratified(var3_copy)
var4_sam, X_test, y_train, y_test = sample_data_stratified(var4_copy)
var5_sam, X_test, y_train, y_test = sample_data_stratified(var5_copy)

# # %%
# ##### Correlation Matrix Heatmap ###
import seaborn as sns
import matplotlib.pyplot as plt

df = [base_copy, var1_copy, var2_copy, var3_copy, var4_copy, var5_copy]
for df in df:
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(35, 35))
    sns.heatmap(
        correlation_matrix,  
        annot=True,          
        cmap='coolwarm',     
        fmt=".2f",          
        linewidths=.5,      
        cbar=True           
    )
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# %%
# #### Distribution of All Columns ###
import matplotlib.pyplot as plt
import seaborn as sns
num_cols = len(base_copy.columns)
x = 4
y = (num_cols + x - 1) // x 
plt.figure(figsize=(x * 5, y * 4))
for i, col in enumerate(base_copy.columns):
    plt.subplot(y, x, i + 1)

    if base_copy[col].nunique() < 5 and base_copy[col].dtype == 'int64':
            sns.countplot(x=col, data=base_copy)
            plt.xlabel(col, fontsize=10)
            plt.ylabel('Count', fontsize=10)

    else:
        sns.histplot(base_copy[col], kde=True, bins=30)
        plt.xlabel(col, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)

plt.tight_layout()
plt.suptitle('All Columns Distribution', y=1.02, fontsize=18) 
plt.show()
# %%
### Modeling ###
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#%%
## data splitting for modeling
def split_train_test(df):
    df.loc[(df['customer_age'] < 50) & (df['fraud_bool'] == 0), 'group'] = 0
    df.loc[(df['customer_age'] < 50) & (df['fraud_bool'] == 1), 'group'] = 1
    df.loc[(df['customer_age'] >= 50) & (df['fraud_bool'] == 0), 'group'] = 2
    df.loc[(df['customer_age'] >= 50) & (df['fraud_bool'] == 1), 'group'] = 3
    df['group'] = df['group'].astype(int)
    X = df.drop(columns=['fraud_bool', 'group'])
    y = df['fraud_bool']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['group'])
    
    return X_train, X_test, y_train, y_test

#%%
x_train_base, x_test_base, y_train_base, y_test_base = split_train_test(base_sam)
x_train_var1, x_test_var1, y_train_var1, y_test_var1 = split_train_test(var1_sam)
x_train_var2, x_test_var2, y_train_var2, y_test_var2 = split_train_test(var2_sam)
x_train_var3, x_test_var3, y_train_var3, y_test_var3 = split_train_test(var3_sam)
x_train_var4, x_test_var4, y_train_var4, y_test_var4 = split_train_test(var4_sam)
x_train_var5, x_test_var5, y_train_var5, y_test_var5 = split_train_test(var5_sam)

# %%
XGB = XGBClassifier()
LGB = LGBMClassifier()
CB = CatBoostClassifier()
GB = GradientBoostingClassifier()

# %%
def k_fold_training_smote(model_instance, X_train, y_train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold, (train_index, val_index) in enumerate(k_fold.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)), # SMOTE 인스턴스
            ('classifier', model_instance)      # 모델 인스턴스 (하이퍼파라미터가 적용된)
        ])
        # 모델 학습
        pipeline.fit(X_train_fold, y_train_fold)
        y_val_proba = pipeline.predict_proba(X_val_fold)[:, 1]
        auc_score = roc_auc_score(y_val_fold, y_val_proba)
        auc_scores.append(auc_score)

    return auc_scores

#%%
default_xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc' # K-Fold 훈련 시에는 'auc'보다 'logloss'가 더 일반적
)
datasets = {
    'base':  (x_train_base,  x_test_base,  y_train_base,  y_test_base),
    'var1':  (x_train_var1,  x_test_var1,  y_train_var1,  y_test_var1),
    'var2':  (x_train_var2,  x_test_var2,  y_train_var2,  y_test_var2),
    'var3':  (x_train_var3,  x_test_var3,  y_train_var3,  y_test_var3),
    'var4':  (x_train_var4,  x_test_var4,  y_train_var4,  y_test_var4),
    'var5':  (x_train_var5,  x_test_var5,  y_train_var5,  y_test_var5)
}
#%% 베이지안 최적화 수행
results = {}

for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
    print(f"\n==== Processing dataset: {name} ====")
    
    xgb_model = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='auc' # K-Fold 평가를 위해 'auc' 사용
    )
    
    auc_scores_cv = k_fold_training_smote(xgb_model, X_tr, y_tr)
    print(f"  K-Fold Cross-Validation AUC scores: {auc_scores_cv}")
    print(f"  Average CV AUC: {np.mean(auc_scores_cv):.4f}")
    print(f"  CV AUC Standard Deviation: {np.std(auc_scores_cv):.4f}")

    final_xgb_model = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='auc'
    )

    final_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', final_xgb_model)
    ])

    final_pipeline.fit(X_tr, y_tr)

    # 중요도 추출 
    booster = final_pipeline.named_steps['classifier'].get_booster()
    importance_dict = booster.get_score(importance_type='weight')  # 'gain', 'cover'도 가능

    # 중요도 DataFrame 생성
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values(by='importance', ascending=False)

    print(importance_df)

    # 매핑 딕셔너리
    fmap = {f"f{i}": col for i, col in enumerate(X_tr.columns)}

    # 변환
    importance_df['feature'] = importance_df['feature'].map(fmap)
    print("변환된 중요도 DataFrame:")
    print(importance_df)
    # 베이지안 최적화된 모델로 예측
    y_pred_label = final_pipeline.predict(X_te)
    y_pred_proba = final_pipeline.predict_proba(X_te)[:, 1]

    auc_score   = roc_auc_score(y_te, y_pred_proba)
    conf_mat    = confusion_matrix(y_te, y_pred_label)
    acc = accuracy_score(y_te, y_pred_label)
    prec = precision_score(y_te, y_pred_label)
    rec = recall_score(y_te, y_pred_label)
    fpr, tpr, thresholds = roc_curve(y_te, y_pred_proba)

    results[name] = {
            'cv_auc':      np.mean(auc_scores_cv),
            'test_auc':    auc_score,
            'confusion':   conf_mat,
            'accuracy':    acc,
            'precision':   prec,
            'recall':      rec,
            'fpr':         fpr,
            'tpr':         tpr,
            'thresholds':  thresholds

        }

    print(f"  테스트 AUC: {auc_score:.4f}, 정확도: {acc:.4f}, 정밀도: {prec:.4f}, 재현율: {rec:.4f}")
#%% 
# [ROC Curve 통합 시각화]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['test_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Variants별 ROC Curve 비교")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

#%%

# 6. 최적의 임계값 찾기 (예시: Youden's J statistic 최대화)
# 'thresholds' 배열은 fpr과 tpr이 계산된 임계값들을 포함합니다.
# thresholds[0]은 실제 사용된 가장 높은 임계값이며, thresholds[-1]은 가장 낮은 임계값입니다.
# 일반적으로 thresholds 배열은 내림차순으로 정렬되어 있습니다.

youden_j_scores = tpr - fpr
optimal_idx = np.argmax(youden_j_scores)
optimal_threshold_youden = thresholds[optimal_idx]
optimal_fpr_youden = fpr[optimal_idx]
optimal_tpr_youden = tpr[optimal_idx]

print(f"\nOptimal Threshold (Youden's J statistic): {optimal_threshold_youden:.4f}")
print(f"  FPR at optimal threshold: {optimal_fpr_youden:.4f}")
print(f"  TPR at optimal threshold: {optimal_tpr_youden:.4f}")

# (0,1) 지점까지의 유클리드 거리 최소화로 최적 임계값 찾기
distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
optimal_idx_distance = np.argmin(distances)
optimal_threshold_distance = thresholds[optimal_idx_distance]

print(f"\nOptimal Threshold (Min distance to (0,1)): {optimal_threshold_distance:.4f}")
print(f"  FPR at optimal threshold: {fpr[optimal_idx_distance]:.4f}")
print(f"  TPR at optimal threshold: {tpr[optimal_idx_distance]:.4f}")

# 선택된 임계값으로 예측 수행
y_pred_custom_threshold = (y_pred_proba >= optimal_threshold_youden).astype(int)
from sklearn.metrics import classification_report
print("\nClassification Report with Optimal Threshold (Youden's J):")
print(classification_report(y_test, y_pred_custom_threshold))
# %%
# 정확도 (Accuracy)
accuracy = accuracy_score(y_test, y_pred_custom_threshold)

# 정밀도 (Precision)
precision = precision_score(y_test, y_pred_custom_threshold, pos_label=1, zero_division=0)

# 재현율 (Recall)
recall = recall_score(y_test, y_pred_custom_threshold, pos_label=1, zero_division=0)

# F1-score (참고용: 정밀도와 재현율의 조화 평균)
f1 = f1_score(y_test, y_pred_custom_threshold, pos_label=1, zero_division=0)

# FPR(False Positive Rate)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_custom_threshold)

# FPR
# 혼동 행렬 계산
# [[TN, FP],
#  [FN, TP]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom_threshold).ravel()
# FPR 계산
if (fp + tn) == 0:
    fpr = 0.0 # 실제 음성 데이터가 없는 경우
else:
    fpr = fp / (fp + tn)
print(f"임계값 {optimal_threshold_youden}에서의 FPR: {fpr:.4f}")
print(f"FP: {fp}, TN: {tn}")

print("\n--- 최적 임계값 적용 시 성능 지표 ---")
print(f"AUC: {auc_score:.4f}")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
# %%
