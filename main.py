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
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = [base_copy, var1_copy, var2_copy, var3_copy, var4_copy, var5_copy]
# for df in df:
#     correlation_matrix = df.corr(numeric_only=True)
#     plt.figure(figsize=(35, 35))
#     sns.heatmap(
#         correlation_matrix,  
#         annot=True,          
#         cmap='coolwarm',     
#         fmt=".2f",          
#         linewidths=.5,      
#         cbar=True           
#     )
#     plt.title('Correlation Matrix Heatmap')
#     plt.show()

# %%
# #### Distribution of All Columns ###
# import matplotlib.pyplot as plt
# import seaborn as sns
# num_cols = len(base_copy.columns)
# x = 4
# y = (num_cols + x - 1) // x 
# plt.figure(figsize=(x * 5, y * 4))
# for i, col in enumerate(base_copy.columns):
#     plt.subplot(y, x, i + 1)

#     if base_copy[col].nunique() < 5 and base_copy[col].dtype == 'int64':
#             sns.countplot(x=col, data=base_copy)
#             plt.xlabel(col, fontsize=10)
#             plt.ylabel('Count', fontsize=10)

#     else:
#         sns.histplot(base_copy[col], kde=True, bins=30)
#         plt.xlabel(col, fontsize=10)
#         plt.ylabel('Frequency', fontsize=10)

# plt.tight_layout()
# plt.suptitle('All Columns Distribution', y=1.02, fontsize=18) 
# plt.show()
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
def k_fold_training(model, X_train, y_train):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold, (train_index, val_index) in enumerate(k_fold.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        auc_score = roc_auc_score(y_val_fold, y_val_proba)
        auc_scores.append(auc_score)

    return auc_scores

#%%
space = [
    Integer(100, 1000, name='n_estimators'), # 트리의 개수
    Real(0.01, 0.2, name='learning_rate', prior='log-uniform'), # 학습률 (로그 스케일이 유리)
    Integer(3, 10, name='max_depth'), # 트리의 최대 깊이
    Real(0.5, 1.0, name='subsample'), # 샘플링 비율 (각 트리마다 사용될 데이터 샘플의 비율)
    Real(0.5, 1.0, name='colsample_bytree'), # 각 트리에 사용할 특성(컬럼) 비율
    Real(1e-9, 10.0, name='reg_alpha', prior='log-uniform'), # L1 정규화
    Real(1e-9, 10.0, name='reg_lambda', prior='log-uniform') # L2 정규화
]
#%%
param_names = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda"
    ]

#%%
def objective(params):
    param_dict = dict(zip(param_names, params))

    model = XGBClassifier(
        **param_dict,
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
        )

    auc_scores = k_fold_training(model, x_train_base, y_train_base)

    return -np.mean(auc_scores)
#%%
res_gp = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    n_random_starts=10,
    random_state=42,
    verbose=True
)

print(f"최적의 ROC AUC (K-Fold 검증 평균): {-res_gp.fun:.4f}")

best_params = {
    'n_estimators': res_gp.x[0],
    'learning_rate': res_gp.x[1],
    'max_depth': res_gp.x[2],
    'subsample': res_gp.x[3],
    'colsample_bytree': res_gp.x[4],
    'reg_alpha': res_gp.x[5],
    'reg_lambda': res_gp.x[6]
}
print("최적의 하이퍼파라미터:")
for name, value in best_params.items():
    print(f"{name}: {value}")
#%%
# 최적의 하이퍼파라미터로 XGBoost 모델 생성
final_best_xgb_model = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
#%%
# 최적의 하이퍼파라미터로 모델 학습
final_best_xgb_model.fit(x_train_base, y_train_base)
# test_y, pred_y를 활용한 지표 적용
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
# 베이지안 최적화된 모델로 예측
y_pred_test_xgb = final_best_xgb_model.predict(x_test_base)[:, 1]
y_pred_test_xgb = final_best_xgb_model.predict_proba(x_test_base)[:, 1]

test_auc_bo_xgb = roc_auc_score(y_test_base, y_pred_test_xgb)
confusion = confusion_matrix(y_test_base, y_pred_test_xgb)
accuracy  = accuracy_score(y_test_base, y_pred_test_xgb)
precision = precision_score(y_test_base, y_pred_test_xgb)
recall    = recall_score(y_test_base, y_pred_test_xgb)

print('================= confusion matrix ====================')
print(confusion)
print('=======================================================')
print(f'정확도:{accuracy}, 정밀도:{precision}, 재현율:{recall}')

print(f"베이지안 최적화된 모델의 최종 테스트 AUC: {test_auc_bo_xgb:.4f}")
