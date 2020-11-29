from pathlib import Path
import sklearn
import numpy as np
import pickle
import argparse
import sys
sys.path.append('src/data/')
from processing_data import get_processing_data
from sklearn.datasets import load_files
import os.path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--DocumentFolder', type=str, required=True, help="Folder of Input Documents")
    args = parser.parse_args()
    path = Path(__file__).parent / f"../../"
    folder_data = load_files(path, categories=args.DocumentFolder)
    data = folder_data.data
    if not data:
        print('Empty Folder')
        exit()
    document = get_processing_data(data)

    tfidf_path = Path(__file__).parent / "../../data/features/tfidf.pickle"
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    doc_vector = tfidf.transform(document).toarray()
    
    svc_path = Path(__file__).parent / "../../model/SVC.pickle"
    with open(svc_path, 'rb') as f:
        svc = pickle.load(f)
    pred = svc.predict(doc_vector)
    category_dict = {0:"Business", 1: 'Entertainment', 2:'Politics', 3:'Sport', 4:'Tech'}
    results = [category_dict[x] for x in pred]
    
    file_name = [os.path.split(x)[-1] for x in folder_data.filenames]
    prediction = zip(file_name, results)
    prediction = list(prediction)
    
    print(prediction)