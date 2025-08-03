# Netflix Movies and TV Shows Clustering

With the vast and continuously expanding catalog of content on Netflix, organizing and understanding this data is essential for both users and content providers. Clustering is an unsupervised machine learning technique that can help uncover hidden patterns in data by grouping similar items together. In this context, **clustering Netflix movies and TV shows** based on various features such as genre, rating, and duration can offer valuable insights into content trends, user preferences, and potential content recommendations.

By analyzing and grouping titles with similar characteristics, we can discover natural groupings that may not be immediately obvious. This approach can also aid in building better recommendation systems, optimizing content curation, and enhancing user experience through personalized content discovery. Additionally, understanding these clusters allows Netflix and other streaming platforms to strategically plan their content offerings based on viewer behavior and preferences.

This project aims to apply clustering techniques such as **K-Means, Hierarchical Clustering, or DBSCAN** to Netflix titles, using key features like genre categories, IMDb or user ratings, and content duration to identify meaningful groupings and patterns within the dataset.
The initial step in our machine learning pipeline involved acquiring and preparing the dataset. The dataset was downloaded from the given source, extracted, and successfully imported into a Pandas DataFrame for further analysis and manipulation.

**Handling Missing and Inconsistent Data** To ensure data quality, we began by inspecting the dataset for null, empty, or anomalous values. This included checking for NaN's and other placeholders that may indicate missing data. Where applicable, imputation techniques were applied to fill in missing values, using methods appropriate to the data type and distribution (e.g., mean or median imputation).

We also performed value counts on individual features to uncover uncommon, unusual, or inconsistent entries. These checks helped in detecting possible data entry errors or formatting issues that could distort model performance. Any inconsistent or unknown values were cleaned or corrected to maintain data integrity.

**Data Formatting and Structure** Next, we standardized column names and formats to ensure uniformity throughout the dataset. This step involved correcting inconsistencies in naming conventions, aligning data types, and organizing the dataset for seamless analysis and model input. 

**Model Training and Recommendation** K-Means, DBSCAN, and Hierarchical Clustering models were trained on the selected input features. The optimal number of clusters was determined using the Elbow Method and Silhouette Score. The resulting clusters were visualized through scatter plots to analyze the cluster formation.

**Cluster Evaluation** Each point on the plot represents a unique movie or TV show, with colors indicating their respective cluster assignments derived from features such as rating, popularity, or other engineered attributes. This clustering reveals patterns in content grouping, enabling more informed and personalized recommendation strategies.
