!pip install tensorflow
!pip install text-preprocessing

import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

import nltk
nltk.download("all")

from text_preprocessing import preprocess_text, remove_number, remove_email, remove_url, remove_whitespace, remove_phone_number, remove_punctuation, remove_special_character, remove_stopword, check_spelling
import contractions

import tensorflow as tf

from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv")
data.head(2)

#Count missing values in each column
data.isnull().sum()

df = data
# Fill missing with 'Unknown' for now
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Unknown', inplace=True)
df['country'].fillna('Unknown', inplace=True)

#errors='coerce' if value it can’t parse into a valid date (e.g., a typo or a null/NaN),
#it should replace it with NaT (Not a Time), instead of throwing an error.
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

mean_date = df['date_added'].mean()
df['date_added_mean'] = df['date_added'].fillna(mean_date)
df.drop('date_added', axis=1, inplace=True)

mode_rating = df['rating'].mode()[0]
df['rating_mode'] = df['rating'].fillna(mode_rating)
df.drop('rating', axis=1, inplace=True)
df.isnull().sum()

df.drop_duplicates(inplace=True)

#Copy Df for analysis
AnalysisData=df

# ****1****  Total Content Types (Movie vs TV Show)
fig = px.histogram(
    AnalysisData,
    x='type',
    color='type',
    title='Distribution of Content Types (Movie vs TV Show)'
)
fig.update_layout(
    xaxis_title='Type',
    yaxis_title='Count',
    width=1000,
    height=400
)
fig.show()

# ****2**** Number of Titles over the years
fig = px.histogram(
    AnalysisData,
    x='release_year',
    nbins=30,
    color='type',
    title='Number of Titles by Release Year'
)
fig.update_layout(
    xaxis_title='Release Year',
    yaxis_title='Count',
    xaxis_tickangle=45
)
fig.show()

# ****3**** Rating Distribution
fig = px.histogram(
    AnalysisData,
    x='rating_mode',
    category_orders={'rating_mode': AnalysisData['rating_mode'].value_counts().index.tolist()},
    title='Distribution of Ratings',
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig.update_layout(
    xaxis_title='Rating',
    yaxis_title='Count',
    xaxis_tickangle=45
)
fig.show()

# Top Genres
from collections import Counter
# Split multiple genres
all_genres = AnalysisData['listed_in'].dropna().str.split(', ').sum()
top_genres = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(10)
# Convert Series to DataFrame for Plotly
top_genres_df = top_genres.reset_index()
top_genres_df.columns = ['Genre', 'Count']
# *****4***** Top 10 Most Common Genres & Create horizontal bar chart
fig = px.bar(
    top_genres_df,
    x='Count',
    y='Genre',
    orientation='h',
    title='Top 10 Most Common Genres',
    color='Count',
    color_continuous_scale='RdBu'
)
fig.update_layout(
    xaxis_title='Count',
    yaxis_title='Genre',
    yaxis=dict(autorange="reversed")  # For descending order
)
fig.show()

# *****5***** Top 10 Countries by Number of Titles
top_countries = AnalysisData['country'].value_counts().head(10)
# Convert Series to DataFrame
top_countries_df = top_countries.reset_index()
top_countries_df.columns = ['Country', 'Count']

# Create Plotly bar chart
fig = px.bar(
    top_countries_df,
    x='Count',
    y='Country',
    orientation='h',
    title='Top 10 Countries by Number of Titles',
    color='Count',
    color_continuous_scale='RdBu'
)
fig.update_layout(
    xaxis_title='Count',
    yaxis_title='Country',
    yaxis=dict(autorange="reversed")
)
fig.show()


# Drop columns that aren't useful or are too descriptive
df = df.drop(columns=['show_id', 'title', 'description', 'cast', 'date_added_mean','duration'])

# Handle missing values if any
df = df.dropna()

# Use LabelEncoder for simplicity
categ_cols = ['type', 'director', 'country', 'listed_in', 'rating_mode']
le = LabelEncoder()
for col in categ_cols:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

#Elbow Method for Optimal Clusters
inertia = []
K = range(1, 21)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

#Draw line chat to find K value
px.line(
    x = range(1,21),
    y = inertia,
    markers = True
)


# The WCSS drops steeply from K=1 to around K=5-6.
# After that the curve bends and starts to flatten — this is the "elbow."
kmeanModel = KMeans(n_clusters=6, random_state=42)
kmeanModel.fit(df)
df['Cluster'] = kmeanModel.predict(df)
df['Cluster'].value_counts()

#Plot to see the clustering
px.scatter(
    data_frame= df,
    color= "Cluster"
)

from sklearn.metrics import silhouette_score
labels = kmeanModel.fit_predict(scaled_features)
# Evaluate with Silhouette Score
score = silhouette_score(scaled_features, labels)
print(f"Silhouette Score: {score}")

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"k={k}, Silhouette Score={score}")


#************************ Hierarchical clustering ***************************************
#***************************************************************

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#***************************************************************
#***************************************************************
#***************************************************************


df.drop(columns='Cluster', inplace=True)
# Perform Hierarchical Clustering
# Create linkage matrix using Ward's method
linkage_matrix = linkage(df, method='ward')

#Plot Dendrogram
#The dendrogram helps decide how many clusters to form by visually cutting the tree
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Assign cluster labels
#The largest height jump appears at the top 2 merges.
#If you draw a horizontal line slightly below the tallest merge (around 140,000 on the Y-axis):
#It will intersect 3 vertical lines, meaning 3 clusters.
df['cluster'] = fcluster(linkage_matrix, t=3, criterion='maxclust')

#Plot to see the clustering
px.scatter(
    data_frame= df,
    color= "cluster"
)

df.drop(columns='cluster', inplace=True)
#************************ DBSCAN clustering ***************************************
#***************************************************************

from sklearn.cluster import DBSCAN

#***************************************************************
#***************************************************************
#***************************************************************

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=3)
clusters = dbscan.fit_predict(df)
# Add cluster labels to DataFrame
df['cluster'] = clusters

#Plot to see the clustering
px.scatter(
    data_frame= df,
    color= "cluster"
)


# Using NLP
#Convert to Lower case
df['listed_in'] = df['listed_in'].str.lower()
df['description'] = df['description'].str.lower()


#Define Preprocess function
def pre_process_function(text):

  text = text.replace("`", "'")
  text = contractions.fix(text)

  text = preprocess_text(text, [remove_url])
  text = preprocess_text(text, [remove_punctuation])
  text = preprocess_text(text, [remove_special_character])
  text = preprocess_text(text, [remove_number])
  text = preprocess_text(text, [remove_whitespace])
  text1 = text
  try:

    text = preprocess_text(text, [check_spelling])
  except:
    text = text1
  text = preprocess_text(text, [remove_stopword])
  return text

df['cleaned_listed_in'] = df['listed_in'].apply(pre_process_function)
df['cleaned_description'] = df['description'].apply(pre_process_function)
df.drop('listed_in', axis=1, inplace=True)
df.drop('description', axis=1, inplace=True)

#Apply TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))

df['combined_text'] = df['cleaned_listed_in'] + " " + df['cleaned_description']

# Fit and transform
X_tfidf = vectorizer.fit_transform(df['combined_text'])

kmeans = KMeans(n_clusters=6, random_state=0)
clusters = kmeans.fit_predict(X_tfidf)
df['cluster'] = clusters

df.drop('cleaned_listed_in', axis=1, inplace=True)
df.drop('cleaned_description', axis=1, inplace=True)
df.drop('combined_text', axis=1, inplace=True)

#Plot to see the clustering
px.scatter(
    data_frame= df,
    color= "cluster"
)
