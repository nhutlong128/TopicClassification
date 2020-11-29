from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def get_features(documents):
    tfidfconverter = TfidfVectorizer(max_features=300, min_df=5, 
                                    max_df=0.7, stop_words=stopwords.words('english'))
    df = tfidfconverter.fit_transform(documents['Content']).toarray()
    feature_names = tfidfconverter.get_feature_names()
    df = pd.DataFrame(df)
    df.columns = feature_names
    df = pd.concat([df, documents['Category']], axis = 1)
    return df


if __name__ == '__main__':
    path = Path(__file__).parent / "../../data/processed/processed.csv"
    documents = pd.read_csv(path)
    df = get_features(documents)
    features_path = Path(__file__).parent / "../../data/features/tfidf_features.csv"
    df.to_csv(path_or_buf = features_path)