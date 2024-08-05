**Importing the necessary libraries**
``
import pandas as pd  # For data manipulation
from sklearn.preprocessing import StandardScaler  # For normalizing the data
from sklearn.cluster import KMeans  # For K-Means clustering
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualization
``

**Loading the dataset**
``
data = pd.read_csv(r'/content/Mall_Customers.csv')
print(data.head())
# Select the relevant columns for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
``

**Normalising the data using scaler object and displaying the scaled data**
``
scaler = StandardScaler()  # Create a scaler object
X_scaled = scaler.fit_transform(X)  # Fit and transform the data
print(X_scaled[:5])
``

**Applying K-Means clustering algorithm by creating a KMeans object with 5 clusters and n_init set to 10 and fitting the model and predicting cluster labels**
``
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
print(data.head())
``

**Visualising the clusters by importing matplotlib.pyplot for plotting**
``
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
``

![image](https://github.com/user-attachments/assets/c3f06aa9-ac52-4a5c-b2f7-f5d5694a68ae)
