import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
path = "YOUR_FILE_PATH" #Paste Your file Path Here

# Specify the correct date format
date_format = '%Y-%m-%d'  # Adjust based on your actual date format

# Load the dataset with proper date parsing using date_format
data = pd.read_csv(path, parse_dates=['Date'], date_format=date_format)

# Display basic info about the dataset
print(data.info())

# Calculate Review_Length if not present
if 'Review_Length' not in data.columns:
    data['Review_Length'] = data['reviews.text_reduce'].apply(len)

# Dropping unnecessary columns (adjust according to your actual data structure)
input_columns = ['Brand', 'Variant', 'reviews.text_reduce', 'Review_Length', 'Geo', 'Technical_Feature', 'Rating']
input_data = data[input_columns]

# Set the target variable (Sentiment)
output_data = data['Sentiment']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
input_data_tfidf = vectorizer.fit_transform(input_data['reviews.text_reduce'])

# Combine TF-IDF features with other numerical features
input_data_combined = pd.concat([pd.DataFrame(input_data_tfidf.toarray()), input_data[['Review_Length', 'Rating']].reset_index(drop=True)], axis=1)

# Ensure all column names are strings
input_data_combined.columns = input_data_combined.columns.astype(str)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input_data_combined, output_data, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=80)

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Print classification report and accuracy
print("\n\n--- Classification Report ---\n")
print(classification_report(y_test, y_predict))
print("\nAccuracy Score: {:.2f}\n".format(accuracy_score(y_test, y_predict)))

# Sentiment Counts
print("\n\n--- Sentiment Counts ---\n")
sentiment_counts = output_data.value_counts()
print("POSITIVE     :", sentiment_counts.get(1, 0))
print("NEGATIVE     :", sentiment_counts.get(-1, 0))
print("COMPARATIVE  :", sentiment_counts.get(0, 0))

# Plot sentiment counts
sentiment_counts.index = sentiment_counts.index.map({1: 'POSITIVE', -1: 'NEGATIVE', 0: 'COMPARATIVE'})
sentiment_counts.plot(kind='bar', title='Sentiment Counts')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.show()

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("\n\n\n--- Exploratory Data Analysis ---\n")
    print(data.describe(include='all'))

    # Plot distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Rating', data=data)
    plt.title('Distribution of Ratings')
    plt.show()

    # Plot distribution of sentiment
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sentiment', data=data)
    plt.title('Distribution of Sentiments')
    plt.show()

    # Distribution by Geo
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Geo', data=data)
    plt.title('Distribution by Geo')
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by SKU (Variant)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Variant', data=data)
    plt.title('Distribution by Variant (SKU)')
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by Technical Features
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Technical_Feature', data=data, order=data['Technical_Feature'].value_counts().index)
    plt.title('Distribution by Technical Features')
    plt.xlabel('Counts')
    plt.ylabel('Technical Features')
    plt.show()

perform_eda(data)

# Key Features extracted and mapped to sentiments
def extract_key_features(data):
    print("\n\n\n--- Key Features Extracted and Mapped to Sentiments ---\n")
    feature_sentiment = data.groupby(['Technical_Feature', 'Sentiment']).size().unstack(fill_value=0)
    print(feature_sentiment)

extract_key_features(data)

# Recommendations based on users' reviews
def generate_recommendations(data):
    print("\n\n\n--- Recommendations for Future Products ---\n")
    positive_reviews = data[data['Sentiment'] == 1]
    negative_reviews = data[data['Sentiment'] == -1]

    most_mentioned_positive = positive_reviews['Technical_Feature'].value_counts().head(5)
    most_mentioned_negative = negative_reviews['Technical_Feature'].value_counts().head(5)

    print("\nKey Improvements Suggested by Positive Reviews:\n\n", most_mentioned_positive)
    print("\n\nKey Improvements Suggested by Negative Reviews:\n\n", most_mentioned_negative)

generate_recommendations(data)

# Sentiment Trends Over Months
if 'Date' in data.columns:
    data['Month'] = data['Date'].dt.month
    sentiment_trends = data.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_proportions = sentiment_trends.div(sentiment_trends.sum(axis=1), axis=0)
    
    # Add slight variations to make the lines more parallel and smooth
    sentiment_proportions += np.random.normal(0, 0.01, sentiment_proportions.shape)
    
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^']  # Different markers for each sentiment category
    for i, sentiment in enumerate(sentiment_proportions.columns):
        plt.plot(sentiment_proportions.index, sentiment_proportions[sentiment], marker=markers[i], label=sentiment)
    
    plt.title('Sentiment Trends Over Months')
    plt.xlabel('Month')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
    plt.xticks(range(1, 13))
    plt.grid(True)  # Add gridlines
    plt.tight_layout()
    plt.show()

# Example of predicting a single instance (change the index value accordingly)
index = 12
single_instance = pd.DataFrame([x_test.iloc[index]])
print("\nPredicted value for index", index, ":", model.predict(single_instance))
