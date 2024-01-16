# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load the dataset
df = pd.read_excel('Online_Retail.xlsx')

# Data Preparation
df.dropna(inplace=True) # Drop null values
df = df[df['Quantity'] > 0] # Keep only positive Quantity values
df['Total_Price'] = df['Quantity'] * df['UnitPrice'] # Calculate Total Price for each transaction

# Calculate Recency, Frequency and Monetary Value for each customer
snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(days=1) # Set snapshot date as one day after the last invoice date
customers = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'count', # Frequency
    'Total_Price': 'sum' # Monetary Value
}).reset_index()

# Rename columns for better understanding
customers.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Total_Price': 'MonetaryValue'}, inplace=True)

# Visualize the distribution of Recency, Frequency and Monetary Value
sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18,5))
sns.histplot(customers['Recency'], ax=axes[0], color='blue')
sns.histplot(customers['Frequency'], ax=axes[1], color='green')
sns.histplot(customers['MonetaryValue'], ax=axes[2], color='purple')
axes[0].set_title('Recency Distribution')
axes[1].set_title('Frequency Distribution')
axes[2].set_title('Monetary Value Distribution')
plt.show()

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers[['Recency', 'Frequency', 'MonetaryValue']])
customers_scaled = pd.DataFrame(customers_scaled, columns=['Recency', 'Frequency', 'MonetaryValue'])

# Visualize the distribution of scaled features
fig, axes = plt.subplots(1, 3, figsize=(18,5))
sns.histplot(customers_scaled['Recency'], ax=axes[0], color='blue')
sns.histplot(customers_scaled['Frequency'], ax=axes[1], color='green')
sns.histplot(customers_scaled['MonetaryValue'], ax=axes[2], color='purple')
axes[0].set_title('Recency Distribution (Scaled)')
axes[1].set_title('Frequency Distribution (Scaled)')
axes[2].set_title('Monetary Value Distribution (Scaled)')
plt.show()

# Evaluate the Model

# Load new customer data
new_data = pd.read_csv('new_customers.csv')

# Preprocess new customer data
new_data_cleaned = preprocess_data(new_data)

# Scale the data using the scaler object from training
new_data_scaled = scaler.transform(new_data_cleaned)

# Use the clustering model to predict segments for new customers
new_customer_segments = kmeans.predict(new_data_scaled)

# Add the predicted segments to the new customer data
new_data['Segment'] = new_customer_segments

# Save the data to a new CSV file
new_data.to_csv('new_customers_segmented.csv', index=False)

# Calculate Silhouette Coefficient for the clustered data
cluster_labels = kmeans.labels_
silhouette_score = metrics.silhouette_score(data_scaled, cluster_labels)

print(f"Silhouette Coefficient for the clustered data: {silhouette_score}")

# Create a scatter plot of the first two principal components, colored by segment
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_)
plt.title('Customer Segments')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
