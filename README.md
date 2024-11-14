# Sentimental-analysis-Intel-Products 2024
A python tool to classify the reviews from E-commerce sites as POSITIVE, NEGATIVE & COMPARATIVE
This project performs sentiment analysis on a dataset of textual reviews. The sentiment of each review is predicted using a Naive Bayes classifier trained on TF-IDF vectorized text data.

# I've completed my project using both Supervised and Unsupervised approaches.

About our files:
Project Structure:-

random_forest.py - supervised approach - RANDOMFOREST APPROACH.

kmeans_approach.py - unsupervised approach - KMEANS APPROACH.

data.csv - Dataset file.

review_scraping.ipynb - used to extract reviews from E-commerce websites.



SETUP:-
Install the libraries using the commands : pip install pandas scikit-learn seaborn matplotlib numpy.


IMPORTANT NOTE:- 
1. Both the codes have the requirement of same csv file ie.,, data.csv.
2. All the outputs are shown in the TERMINAL in VSCODE.

Output:
The script prints the following information:

- Classification Report.
- Accuracy Score.
- Sentiment Counts(Total number of different reviews present in the csv file).
- GRAPHS - 1. Sentimental Counts.
- 2. Distribution of Ratings, Sentiments, Geo, SKU, Technical features.
- Sentiment trends over a period.
- Exploratory Data Analysis.
- Key Features Extracted and Mapped to Sentiments.
- Key Improvements Suggested by Positive and Negative Reviews.



