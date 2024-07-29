#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('Downloads\\Mall_Customers.csv')


# In[4]:


print(df.head())
print(df.info())
print(df.describe())


# In[5]:


from sklearn.preprocessing import StandardScaler

# Select relevant features for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


# In[7]:


kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


# In[8]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()


# In[9]:


cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = ['Cluster ' + str(i) for i in range(5)]
cluster_data = pd.DataFrame(cluster_centers, columns=features)
cluster_data['Cluster'] = cluster_labels

print(cluster_data)


# In[ ]:




