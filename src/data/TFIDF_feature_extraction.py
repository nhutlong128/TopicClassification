from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import argparse


def get_features(documents, max_features, min_df, max_df):
    tfidfconverter = TfidfVectorizer(max_features=max_features, min_df=min_df, 
                                    max_df=max_df, stop_words=stopwords.words('english'))
    df = tfidfconverter.fit_transform(documents['Content']).toarray()
    feature_names = tfidfconverter.get_feature_names()
    df = pd.DataFrame(df)
    df.columns = feature_names
    df = pd.concat([df, documents['Category']], axis = 1)
    return df, tfidfconverter


def formalize_argument(args):
    if args.MaxFeatures is None:
        args.MaxFeatures = 300
    if args.MinDF is None:
        args.MinDF = 5
    if args.MaxDF is None:
        args.MaxDF = 0.7
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", '--MaxFeatures', type=int, help="build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.")
    parser.add_argument("-i", '--MinDF', type=float, help="When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts.")
    parser.add_argument("-a", '--MaxDF', type=float, help="When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts")
    args = parser.parse_args()
    path = Path(__file__).parent / "../../data/processed/processed.csv"
    documents = pd.read_csv(path)
    args = formalize_argument(args)
    df, tfidf = get_features(documents, args.MaxFeatures, args.MinDF, args.MaxDF)
    print(f'Max Features : {args.MaxFeatures}')
    print(f'Min DF : {args.MinDF}')
    print(f'Max DF : {args.MaxDF}')
    features_path = Path(__file__).parent / "../../data/features/tfidf_features.csv"
    df.to_csv(path_or_buf = features_path)
    tfidf_path = Path(__file__).parent / "../../data/features/tfidf.pickle"
    pickle.dump(tfidf, open(tfidf_path, "wb"))
    print('TFIDF Vectorizer saved')
