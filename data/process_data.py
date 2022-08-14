import sys
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the message and category data functions
    
    Arguments:
        messages_filepath -> where messages csv file is located
        categories_filepath -> where categories csv file is located
    Output:
        df -> Combined data containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = ['id'])  
    return df
    


def clean_data(df):
    """
    - Cleans the combined message and category dataframe
    
    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame
    """
    categories = df.categories.str.split(pat=';',expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames = firstrow.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    # Removing entry that is non-binary in the related column
    df.related.replace(2,1,inplace=True)
    df = df.drop_duplicates()
    return df
        


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageTable', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()