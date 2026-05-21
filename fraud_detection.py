import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# SECTION 3 - DATA PREPARATION

# Step 1 - Load the dataset
df = pd.read_csv('creditcard.csv')
print("Dataset loaded successfully")
print("Shape:", df.shape)

# Step 2 - Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 3 - Check class distribution
print("\nClass distribution:")
print(df['Class'].value_counts())

# Step 4 - Check for duplicates
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)

# Step 5 - Scale Amount and Time columns
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
print("\nAmount and Time columns scaled successfully")

# Step 6 - Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Step 7 - Split into 80% training and 20% testing
# stratify=y makes sure both splits have the same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain/test split done")
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# Step 8 - Apply SMOTE to training data only
# This balances the classes by generating synthetic fraud examples
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("\nSMOTE applied successfully")
print("Training size after SMOTE:", X_train_res.shape)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# SECTION 4 - MODELLING, PREDICTION AND RECOMMENDATIONS

# Step 9 - Train Logistic Regression as baseline comparison
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_lr))

# Step 10 - Train Random Forest (main model)
print("\nTraining Random Forest (this may take a few minutes)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_rf))

# Step 11 - Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png")

# Step 12 - Feature Importance Chart
feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='steelblue')
plt.title('Top 10 Most Important Features - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Feature importance chart saved as feature_importance.png")

# Step 13 - Class imbalance visualisation
plt.figure(figsize=(6, 4))
df['Class'].value_counts().plot(kind='bar', 
                                 color=['steelblue', 'red'])
plt.title('Class Distribution - Legitimate vs Fraud')
plt.xticks([0, 1], ['Legitimate', 'Fraud'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Class distribution chart saved as class_distribution.png")

print("\nAll done! Check your folder for the saved graphs")