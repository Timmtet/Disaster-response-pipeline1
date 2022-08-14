import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def load_data(database_filepath):
    """
    Loads data from SQL Database
    Args:
    database_filepath: Path to SQL database file
    Returns:
    X -> a dataframe containing features
    Y -> a dataframe containing labels
    category_names list: Target labels 
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageTable', con=engine)
    #df = pd.read_sql("SELECT * FROM MessageTable", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    
    """
    Normalize, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = stopwords.words("english")
    
    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [20, 50]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv

   


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Displays model's performance on test data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    
    
    y_pred = model.predict(X_test)
    

    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i], target_names=category_names))
    
    


def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()