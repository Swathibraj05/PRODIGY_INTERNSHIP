**Importing the necessary libraries**

```python

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns 
```

**Loading the dataset**

```python

data = pd.read_csv(r'/content/Mall_Customers.csv')
print(data.head())
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
```

![image](https://github.com/user-attachments/assets/95d0e9e3-4f4a-4f5f-b312-30d2973e1240)


**Normalising the data using scaler object and displaying the scaled data**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])
```

![image](https://github.com/user-attachments/assets/97e4ec21-6a2c-4101-bd23-efe56e36fe6c)


**Applying K-Means clustering algorithm by creating a KMeans object with 5 clusters and n_init set to 10 and fitting the model and predicting cluster labels**

```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
print(data.head())
```

![image](https://github.com/user-attachments/assets/0d7c79a1-d8ae-4025-8f0b-f3ff00f12867)


**Visualising the clusters by importing matplotlib.pyplot for plotting**

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

![image](https://github.com/user-attachments/assets/c3f06aa9-ac52-4a5c-b2f7-f5d5694a68ae)
