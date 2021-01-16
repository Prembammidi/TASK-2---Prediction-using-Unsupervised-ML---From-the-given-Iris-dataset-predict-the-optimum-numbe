# TASK-2---Prediction-using-Unsupervised-ML---From-the-given-Iris-dataset-predict-the-optimum-numbe

BAMMIDI PREM KUMAR
DATA SCIENCE AND BUSINESS ANALYTICS
GRIP - THE SPARKS FOUNDATION
TASK2 : Prediction using Unsupervised ML - From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# import all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# datasets for prediction
​
​
iris_df = pd.read_csv("Iris.csv")
iris_df
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa
...	...	...	...	...	...	...
145	146	6.7	3.0	5.2	2.3	Iris-virginica
146	147	6.3	2.5	5.0	1.9	Iris-virginica
147	148	6.5	3.0	5.2	2.0	Iris-virginica
148	149	6.2	3.4	5.4	2.3	Iris-virginica
149	150	5.9	3.0	5.1	1.8	Iris-virginica
150 rows × 6 columns

iris_df.head()
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa
# K MEANS CLASSIF AND PLOTTING THE RESULTS
​
q = iris_df.iloc[:, [0, 1, 2, 3]].values
​
from sklearn.cluster import KMeans
cls = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                   max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(q)
    cls.append(kmeans.inertia_)
    
plt.plot(range(1, 11), cls)
plt.title('The allow method')
plt.xlabel('Number of clusters')
plt.ylabel('cls')
plt.show()

# apply Kmeans to dataset 
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(q)
kmeans.cluster_centers_
array([[ 21.5       ,   5.21428571,   3.62142857,   1.50714286],
       [111.        ,   6.65294118,   2.95294118,   5.70588235],
       [ 63.        ,   5.81333333,   2.70666667,   4.16      ],
       [142.5       ,   6.49375   ,   3.03125   ,   5.39375   ],
       [ 94.5       ,   5.7875    ,   2.7625    ,   4.2875    ],
       [ 49.        ,   5.50769231,   3.23846154,   2.67692308],
       [ 78.5       ,   6.05      ,   2.81875   ,   4.4       ],
       [  7.5       ,   4.85      ,   3.3       ,   1.43571429],
       [ 35.5       ,   5.00714286,   3.32142857,   1.42142857],
       [127.        ,   6.68666667,   2.93333333,   5.54666667]])
# visual of clusters
​
plt.scatter(q[y_kmeans == 0, 0], q[y_kmeans == 0, 1],
            s = 100, c = 'green', label = 'Iris-setosa')
plt.scatter(q[y_kmeans == 1, 0], q[y_kmeans == 1, 1],
           s = 100, c = 'red', label = 'Iris-versicolour')
plt.scatter(q[y_kmeans == 2, 0], q[y_kmeans == 2, 1],
           s = 100, c = 'yellow', label = 'Iris-virginica')
​
            
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'blue', label = 'centroids')
            
plt.legend()
<matplotlib.legend.Legend at 0x20f8acdd430>
