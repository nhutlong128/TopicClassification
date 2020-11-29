from pathlib import Path
import sklearn
import numpy as np
import pickle




if __name__ == '__main__':
    document = 'tate lyle bos bag top award r tate lyle chief executive ha been named european businessman of the year by leading business magazine r iain ferguson wa awarded the title by u publication forbes for returning one of the uk venerable manufacturer to the country top 100 company the sugar group had been absent from the ftse 100 for seven year until mr ferguson helped it return to growth tate share have leapt 55 this year boosted by firming sugar price and sale of it artificial sweetener r after year of sagging stock price and seven year hiatus from the ftse 100 one of britain venerable manufacturer ha returned to the vaunted index forbes said mr ferguson took the helm at the company in 2003 after spending most of his career at consumer good giant unilever tate lyle which wa an original member of the historic ft 30 index in 1935 operates more than 41 factory and 20 more additional production facility in 28 country previous winner of the forbes award include royal bank of scotland'
    tfidf_path = Path(__file__).parent / "../../data/features/tfidf.pickle"
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    doc_vector = tfidf.transform([document]).toarray()
    
    svc_path = Path(__file__).parent / "../../model/SVC.pickle"
    with open(svc_path, 'rb') as f:
        svc = pickle.load(f)
    pred = svc.predict(doc_vector)
    category_dict = {0:"Business", 1: 'Entertainment', 2:'Politics', 3:'Sport', 4:'Tech'}
    
    print(category_dict[pred[0]])