import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import xgboost as xgb

# Loading data into dataframe credit_df
credit_df = pd.read_csv('credit_risk_dataset.csv')

# Cleaning data - removing all rows with NaN values
credit_df = credit_df.dropna()

ord_encode = OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']])
credit_df['loan_grade'] = ord_encode.fit_transform(credit_df['loan_grade'].values.reshape(-1, 1)) # changes ABCDEFG to 0123456
credit_df['person_home_ownership'] = LabelEncoder().fit_transform(credit_df['person_home_ownership'])
credit_df['loan_intent'] = LabelEncoder().fit_transform(credit_df['loan_intent'])
credit_df['cb_person_default_on_file'] = LabelEncoder().fit_transform(credit_df['cb_person_default_on_file'])


# Splitting input data and target data
X =  credit_df.drop(columns='loan_status') # input data (all columns but loan_status)
Y =  credit_df['loan_status'].values # target data (loan_status)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

# LOGISTIC REGRESSION 
model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

auc_score = roc_auc_score(Y_test, y_pred)
print(f"AUC-ROC score: {auc_score:.2f}")

accuracy = accuracy_score(Y_test, y_pred) # accuracy = (TP + TN) / Total
precision = precision_score(Y_test, y_pred) # precision = TP / (TP + FP)
recall = recall_score(Y_test, y_pred) # recall = TP / (TP + FN)
f1 = f1_score(Y_test, y_pred) # 2*(precision*recall)/(precision+recall)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
#print(classification_report(Y_test, y_pred))

# XGB
train_matrix = xgb.DMatrix(X_train, label=Y_train)
test_matrix = xgb.DMatrix(X_test, label=Y_test)

params = {
    'object': 'binary:logistic',
    'eval_metric': 'logloss'
}

model = xgb.train(params, train_matrix, 90)
Y_pred = model.predict(test_matrix)
Y_pred_labels = [1 if p > 0.5 else 0 for p in Y_pred]

accuracy = accuracy_score(Y_test, Y_pred_labels)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(Y_test, Y_pred_labels))

auc_score = roc_auc_score(Y_test, Y_pred)
print(f"AUC-ROC score: {auc_score:.2f}")




#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

#for train_i, test_i in skf.split(X,Y):
#    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
#    Y_train, Y_test = Y[train_i], Y[test_i]
#    print("TRAIN:", train_i, "TEST:", test_i)
#    print("X_train:", X_train, "X_test:", X_test)
#    print("y_train:", Y_train, "y_test:", Y_test)






