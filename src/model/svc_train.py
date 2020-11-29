import sklearn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from model_report import get_tfidf_features
from sklearn.model_selection import train_test_split
import argparse


def fit_evaluate_model(model, x_train, x_valid, x_test, y_train, y_valid, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    train_score = f1_score(y_train, pred, average='macro')
    pred = model.predict(x_valid)
    valid_score = f1_score(y_valid, pred, average='macro')
    pred = model.predict(x_test)
    test_score = f1_score(y_test, pred, average='macro')
    return model, train_score, valid_score, test_score


def formalize_argument(args):
    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
    if args.Kernel not in kernel_list:
        print(f'{args.Kernel} is not in the Kernel list. So we change to default value ~ rbf')
        args.Kernel = 'rbf'
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--C', type=float, default=1, help="Regularization parameter")
    parser.add_argument("-k", '--Kernel', type=str, default='rbf', help="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’")
    parser.add_argument("-d", '--Degree', type=int, default=3, help="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.")
    parser.add_argument("-s", '--SaveModel', type=bool, default=True, help="Save this model or not")
    args = parser.parse_args()
    args = formalize_argument(args)
    

    model = SVC(C=args.C, kernel=args.Kernel, degree = args.Degree)
    x, y = get_tfidf_features()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=6, stratify=y)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    model, train_score, valid_score, test_score = fit_evaluate_model(model, x_train, x_valid, x_test, y_train, y_valid, y_test)
    print(train_score, valid_score, test_score)
    if args.SaveModel:
        path = Path(__file__).parent / "../../model/SVC.pickle"
        pickle.dump(model, open(path, "wb"))
        print('Model saved')



