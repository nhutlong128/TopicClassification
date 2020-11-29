# TopicClassification
# What is Topic Classification?
Input: A corpus.
Output: A topic of the above corpus. Now we are supporting Business, Entertainment, Politics, Sport, Tech topics.

# How to get started:
1. Clone this project: 
git clone https://github.com/nhutlong128/TopicClassification.git
or download this project as zip.
2. Change the working directory to TopicClassification using cd 
3. Create a virtual Python Environment: python -m venv venv_folder
4. Activate the above virtual Environment:
Window: venv_folder\Scripts\Activate.bat
Linux: source venv_folder/Scripts/Activate
5. Install all the requirements library, package by using pip: pip install -r requirements.txt
6. Run setup.py to install necessary nltk packages: python setup.py

# Folder Structure:
1.data:
 - data/raw: Store any raw corpus. Each directory is named after a topic and all text files (.txt) in these directories belong to its parent folder name which is a corresponding topic.
 - data/processed: Store a csv file having two columns 'Content' and 'Category'. Content column contains all the corpus has been preprocessed, Category columns contains these label, topic of these corpus.
 - data/features: Store a csv file which is a feature table exported from TF_IDF.
2.model: containing the model has been trained (fit to the training set of the dataset) which also has the best performance (f1_score). Till now, it's is SVC (Support Vector Machine)
3.report: containing a csv file having performance of all baseline classifier models. Baseline meaning using the default parameter of these.
4.src: containing scripts.
 - src/data: containg two processing data scripts. processing-data.py is use for preprocessing, formalizing a raw corpus. TFIDF_feature_extraction is use for feature extraction using TF_IDF
 - src/model: model_report is to get a report of the performances of all baseline classifier models. svc_train is use to train a SVC model with the specific parameters. svc_predict is to predict new corpus.
5.venv: containing virtual environment parts.
  
