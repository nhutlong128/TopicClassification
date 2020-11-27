from pathlib import Path
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

path = Path(__file__).parent / "../../data/raw"

news_data = load_files(path)
x, y = news_data.data, news_data.target
documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(x)):
    document = str(x[sen]).replace('\\n', ' ')
    
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)


df = pd.DataFrame()
df['Content'] = documents
df['Category'] = news_data.target
output = Path(__file__).parent / "../../data/processed/processed.csv"
df.to_csv(path_or_buf = output)

