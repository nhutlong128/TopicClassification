import sklearn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def get_tfidf_features(): 
    path = Path(__file__).parent / "../../data/features/tfidf_features.csv"
    df = pd.read_csv(path, index_col=0)
    x = df.drop(columns=['Category'])
    y = df['Category']
    return x, y


def get_fold_result(model, x, y):
    name = model[1]
    model = model[0]
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(x, y)
    store = []
    for train_idx, test_idx in skf.split(x, y):
        x_train_fold, x_test_fold = x.loc[train_idx, :], x.loc[test_idx, :]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        model.fit(x_train_fold, y_train_fold)
        pred = model.predict(x_test_fold)
        score = f1_score(y_test_fold, pred, average='macro')
        store.append(score)
    return store, np.mean(store), np.std(store)


if __name__ ==  "__main__":
    dict = {'dt':[DecisionTreeClassifier(), 'Decision Tree Classifier'], 
            'rf':[RandomForestClassifier(random_state=42), 'Random Forest Classifier'], 
            'svc':[SVC(), 'Support Vector Machine Classifier'], 
            'nb':[MultinomialNB(), 'Multinomial Naive Bayes'], 
            'xg':[XGBClassifier(objective='multi:softmax', num_class=5, random_state=42), 'XGBoost Classifier'],
            'lg':[LGBMClassifier(random_state=42), 'LightGBM Classifier']}
    x, y = get_tfidf_features()
    arr_result = []
    for k, v in dict.items():
        temp = []
        score, mean, std = get_fold_result(v, x, y)
        score.extend([mean, std])
        temp.extend([v[1]])
        temp.extend(score)
        arr_result.append(temp)
    
    report = pd.DataFrame(arr_result, columns=['Model', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4',
                        'Fold 5', 'Mean', 'STD'])
    report_path = Path(__file__).parent / "../../report/report.csv"
    report.to_csv(path_or_buf = report_path)



