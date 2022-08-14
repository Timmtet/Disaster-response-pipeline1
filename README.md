Introduction

In this project, a model that classifies messages that are sent during disasters was built. There are 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project involved the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories. The dataset was provided by Figure Eight containing real messages that were sent during disaster events.

Finally, this project contains a web app where you can input a message and get classification results.


Project Dependencies:

Python 3.5+
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Data Visualization: Flask, Plotly
  

Workspace Content:

app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to

|- MessageDatabase

models

|- train_classifier.py

Disaster project1.jpeg

Disaster project2.jpeg

README.md

1. Data:

a) process_data.py: reads in the data, cleans and stores it in a SQL database.
b) disaster_categories.csv and disaster_messages.csv (dataset)
c) DisasterResponse.db: created database from transformed and cleaned data.

2. Models: train_classifier.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. 

3. App: 

a) run.py: Flask app and the user interface used to predict results and display them.
b) templates: folder containing the html templates


Instructions:

Run the following commands in the project's root directory to set up your database and model.

1. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/


Licensing, Authors, Acknowledgements:

Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.