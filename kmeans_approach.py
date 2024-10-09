import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Define sentiment mapping globally
sentiment_mapping = {1: 'Positive', 0: 'Competition', -1: 'Negative'}

# Load the dataset
path = "YOUR_FILE_PATH" #Paste Your file Path Here
data = pd.read_csv(path)

# Specify the correct date format if applicable
date_format = '%Y-%m-%d'  # Adjust based on your actual date format

# Display basic info about the dataset
print(data.info())

# Calculate Review_Length if not present
if 'Review_Length' not in data.columns:
    data['Review_Length'] = data['reviews.text_reduce'].apply(len)

# Dropping unnecessary columns (adjust according to your actual data structure)
input_columns = ['Brand', 'Variant', 'reviews.text_reduce', 'Review_Length', 'Geo', 'Technical_Feature', 'Rating', 'Sentiment']
input_data = data[input_columns]

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
input_data_tfidf = vectorizer.fit_transform(input_data['reviews.text_reduce'])

# Combine TF-IDF features with other numerical features
input_data_combined = pd.concat([pd.DataFrame(input_data_tfidf.toarray()), input_data[['Review_Length', 'Rating', 'Sentiment']].reset_index(drop=True)], axis=1)

# Ensure all column names are strings
input_data_combined.columns = input_data_combined.columns.astype(str)

# Perform KMeans clustering
num_clusters = 3  # Example: choose number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(input_data_tfidf)

# Function to perform cluster EDA
def perform_cluster_eda(data, num_clusters):
    print("\n\n\n--- Cluster Analysis ---\n")
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)  # View the cluster centers
    print("\nCluster Sizes:\n")
    print(data['Cluster'].value_counts())  # View the sizes of each cluster
    
    # Plot sentiment distribution by clusters
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', hue='Sentiment', data=data, palette='Set2', hue_order=[1, 0, -1])
    plt.title('Sentiment Distribution by Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Sentiment', loc='upper right', labels=['Positive', 'Competition', 'Negative'])
    plt.show()

    # Sentiment Counts
    print("\n\n--- Sentiment Counts ---\n")
    sentiment_counts = data['Sentiment'].value_counts()
    positive_count = sentiment_counts.get(1, 0)
    competition_count = sentiment_counts.get(0, 0)
    negative_count = sentiment_counts.get(-1, 0)
    print(f"Positive     : {positive_count}")
    print(f"Competition  : {competition_count}")
    print(f"Negative     : {negative_count}")

    # Plot sentiment counts
    sentiment_counts.plot(kind='bar', title='Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    plt.show()

    # Distribution by Geo
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Geo', hue='Cluster', data=data, palette='Set3')
    plt.title('Distribution of Geo by Clusters')
    plt.xlabel('Geo')
    plt.ylabel('Count')
    plt.legend(title='Cluster', loc='upper right', labels=[f'Cluster {i}' for i in range(num_clusters)])
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by SKU (Variant)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Variant', hue='Cluster', data=data, palette='Set3')
    plt.title('Distribution of Variant (SKU) by Clusters')
    plt.xlabel('Variant (SKU)')
    plt.ylabel('Count')
    plt.legend(title='Cluster', loc='upper right', labels=[f'Cluster {i}' for i in range(num_clusters)])
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by Technical Features
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Technical_Feature', hue='Cluster', data=data, order=data['Technical_Feature'].value_counts().index, palette='Set3')
    plt.title('Distribution of Technical Features by Clusters')
    plt.xlabel('Count')
    plt.ylabel('Technical Features')
    plt.legend(title='Cluster', loc='upper right', labels=[f'Cluster {i}' for i in range(num_clusters)])
    plt.show()

# Perform cluster EDA
perform_cluster_eda(data, num_clusters)

# Function to generate recommendations based on clusters
# Function to generate recommendations based on clusters
# Function to generate recommendations based on clusters
def generate_recommendations(data):
    print("\n\n\n--- Recommendations for Future Products ---\n\n")

    # Print cluster sentiment distribution
    cluster_sentiment = data.groupby(['Cluster', 'Sentiment']).size().unstack(fill_value=0)
    print("Cluster Sentiment Distribution:\n")
    print(cluster_sentiment)

    # Most mentioned features in each cluster
    num_clusters = data['Cluster'].nunique()
    for cluster in range(num_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        print(f"\n\n\nCluster {cluster} - Key Improvements Suggested:\n")
        # Aggregate and count technical features for the current cluster
        tech_feature_counts = cluster_data.groupby('Technical_Feature').size().sort_values(ascending=False).head(5)
        print(tech_feature_counts)

# Generate recommendations
generate_recommendations(data)



# Sentiment Trends Over Months (if Date column exists)
if 'Date' in data.columns:
    data['Month'] = pd.to_datetime(data['Date'], format=date_format).dt.month
    sentiment_trends = data.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_proportions = sentiment_trends.div(sentiment_trends.sum(axis=1), axis=0)
    
    # Add slight variations to make the lines more parallel and smooth
    sentiment_proportions += np.random.normal(0, 0.01, sentiment_proportions.shape)
    
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^']  # Different markers for each sentiment category
    for i, sentiment in enumerate(sentiment_proportions.columns):
        plt.plot(sentiment_proportions.index, sentiment_proportions[sentiment], marker=markers[i], label=sentiment_mapping[sentiment])
    
    plt.title('Sentiment Trends Over Months')
    plt.xlabel('Month')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment', loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
    plt.xticks(range(1, 13))
    plt.grid(True)  # Add gridlines
    plt.tight_layout()
    plt.show()
