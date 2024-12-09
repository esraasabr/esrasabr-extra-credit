# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier  # Requires installation of xgboost

# Load datasets
train = pd.read_csv("/Users/esraasabr/Desktop/506/Extra Credit Competition/train (2).csv")
test = pd.read_csv("/Users/esraasabr/Desktop/506/Extra Credit Competition/test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Feature engineering
def feature_engineering(df):
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['trans_time'] = pd.to_datetime(df['trans_time'], errors='coerce', format='%H:%M:%S')

    df['hour'] = df['trans_time'].dt.hour
    df['day'] = df['trans_date'].dt.dayofweek
    df['is_weekend'] = df['day'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    df['amt_log'] = np.log1p(df['amt'])
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Encode categorical variables
categorical_features = ['category', 'gender', 'state', 'job']
for col in categorical_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Drop unused columns
unused_columns = ['trans_num', 'first', 'last', 'dob', 'merchant', 'street', 'city', 'zip', 'trans_date', 'trans_time']
X = train.drop(['is_fraud'] + unused_columns, axis=1)
y = train['is_fraud']
X_test = test.drop(unused_columns, axis=1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define base models
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
xgb = XGBClassifier(n_estimators=100, max_depth=10, random_state=42, use_label_encoder=False, eval_metric='logloss')
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

# Define meta-model
meta_model = LogisticRegression(max_iter=1000)

# Define StackingClassifier
stacking_model = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gbm', gbm)],
    final_estimator=meta_model,
    cv=3,
    n_jobs=-1
)

# Train the StackingClassifier
stacking_model.fit(X_train, y_train)

# Validate the StackingClassifier
y_val_pred = stacking_model.predict(X_val)
f1 = f1_score(y_val, y_val_pred)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation F1-score: {f1}")
print(f"Validation Accuracy: {accuracy}")

# Predict on the test set using the StackingClassifier
test_predictions = stacking_model.predict(X_test)

# Prepare submission file
sample_submission['is_fraud'] = test_predictions
sample_submission.to_csv("submission_stacking.csv", index=False)
print("Ensemble submission file created: submission_stacking.csv")



