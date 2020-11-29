import sklearn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from model_report import get_tfidf_features
from sklearn.model_selection import train_test_split

def fit_evaluate_model(model, x_train, x_valid, x_test, y_train, y_valid, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    train_score = f1_score(y_train, pred, average='macro')
    pred = model.predict(x_valid)
    valid_score = f1_score(y_valid, pred, average='macro')
    pred = model.predict(x_test)
    test_score = f1_score(y_test, pred, average='macro')
    return model, train_score, valid_score, test_score

if __name__ == '__main__':
    model = SVC()
    x, y = get_tfidf_features()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=6, stratify=y)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    model, train_score, valid_score, test_score = fit_evaluate_model(model, x_train, x_valid, x_test, y_train, y_valid, y_test)
    print(train_score, valid_score, test_score)
    path = Path(__file__).parent / "../../model/SVC.pickle"
    pickle.dump(model, open(path, "wb"))
    print('Model saved')



