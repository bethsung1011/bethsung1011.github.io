---
layout: single
title:  "Precourse Algebra Example 1!"
---

# It is about Matrix, Dot Product and Cosine Similarities.


```python
# Imports
from sklearn.datasets import load_iris

import numpy as np

# Read in data set

iris_dataset = load_iris()
iris_dataset
```




    {'data': array([[5.1, 3.5, 1.4, 0.2],
            [4.9, 3. , 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5. , 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5. , 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3. , 1.4, 0.1],
            [4.3, 3. , 1.1, 0.1],
            [5.8, 4. , 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.5, 0.4],
            [4.6, 3.6, 1. , 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5. , 3. , 1.6, 0.2],
            [5. , 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2],
            [5. , 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.6, 1.4, 0.1],
            [4.4, 3. , 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5. , 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5. , 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3. , 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5. , 3.3, 1.4, 0.2],
            [7. , 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4. , 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1. ],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
            [5. , 2. , 3.5, 1. ],
            [5.9, 3. , 4.2, 1.5],
            [6. , 2.2, 4. , 1. ],
            [6.1, 2.9, 4.7, 1.4],
            [5.6, 2.9, 3.6, 1.3],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3. , 4.5, 1.5],
            [5.8, 2.7, 4.1, 1. ],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1],
            [5.9, 3.2, 4.8, 1.8],
            [6.1, 2.8, 4. , 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [6.1, 2.8, 4.7, 1.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.6, 3. , 4.4, 1.4],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3. , 5. , 1.7],
            [6. , 2.9, 4.5, 1.5],
            [5.7, 2.6, 3.5, 1. ],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.4, 3.7, 1. ],
            [5.8, 2.7, 3.9, 1.2],
            [6. , 2.7, 5.1, 1.6],
            [5.4, 3. , 4.5, 1.5],
            [6. , 3.4, 4.5, 1.6],
            [6.7, 3.1, 4.7, 1.5],
            [6.3, 2.3, 4.4, 1.3],
            [5.6, 3. , 4.1, 1.3],
            [5.5, 2.5, 4. , 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3. , 4.6, 1.4],
            [5.8, 2.6, 4. , 1.2],
            [5. , 2.3, 3.3, 1. ],
            [5.6, 2.7, 4.2, 1.3],
            [5.7, 3. , 4.2, 1.2],
            [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.9, 4.3, 1.3],
            [5.1, 2.5, 3. , 1.1],
            [5.7, 2.8, 4.1, 1.3],
            [6.3, 3.3, 6. , 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3. , 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3. , 5.8, 2.2],
            [7.6, 3. , 6.6, 2.1],
            [4.9, 2.5, 4.5, 1.7],
            [7.3, 2.9, 6.3, 1.8],
            [6.7, 2.5, 5.8, 1.8],
            [7.2, 3.6, 6.1, 2.5],
            [6.5, 3.2, 5.1, 2. ],
            [6.4, 2.7, 5.3, 1.9],
            [6.8, 3. , 5.5, 2.1],
            [5.7, 2.5, 5. , 2. ],
            [5.8, 2.8, 5.1, 2.4],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 3. , 5.5, 1.8],
            [7.7, 3.8, 6.7, 2.2],
            [7.7, 2.6, 6.9, 2.3],
            [6. , 2.2, 5. , 1.5],
            [6.9, 3.2, 5.7, 2.3],
            [5.6, 2.8, 4.9, 2. ],
            [7.7, 2.8, 6.7, 2. ],
            [6.3, 2.7, 4.9, 1.8],
            [6.7, 3.3, 5.7, 2.1],
            [7.2, 3.2, 6. , 1.8],
            [6.2, 2.8, 4.8, 1.8],
            [6.1, 3. , 4.9, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [7.2, 3. , 5.8, 1.6],
            [7.4, 2.8, 6.1, 1.9],
            [7.9, 3.8, 6.4, 2. ],
            [6.4, 2.8, 5.6, 2.2],
            [6.3, 2.8, 5.1, 1.5],
            [6.1, 2.6, 5.6, 1.4],
            [7.7, 3. , 6.1, 2.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 3.1, 5.5, 1.8],
            [6. , 3. , 4.8, 1.8],
            [6.9, 3.1, 5.4, 2.1],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3],
            [5.8, 2.7, 5.1, 1.9],
            [6.8, 3.2, 5.9, 2.3],
            [6.7, 3.3, 5.7, 2.5],
            [6.7, 3. , 5.2, 2.3],
            [6.3, 2.5, 5. , 1.9],
            [6.5, 3. , 5.2, 2. ],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3. , 5.1, 1.8]]),
     'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
     'frame': None,
     'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
     'DESCR': '.. _iris_dataset:\n\nIris plants dataset\n--------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n                \n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThe famous Iris database, first used by Sir R.A. Fisher. The dataset is taken\nfrom Fisher\'s paper. Note that it\'s the same as in R, but not as in the UCI\nMachine Learning Repository, which has two wrong data points.\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\n.. topic:: References\n\n   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...',
     'feature_names': ['sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)'],
     'filename': '/home/beth/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv'}




```python

iris_vectors = iris_dataset['data']

iris_vectors
```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2],
           [5.4, 3.9, 1.7, 0.4],
           [4.6, 3.4, 1.4, 0.3],
           [5. , 3.4, 1.5, 0.2],
           [4.4, 2.9, 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.1],
           [5.4, 3.7, 1.5, 0.2],
           [4.8, 3.4, 1.6, 0.2],
           [4.8, 3. , 1.4, 0.1],
           [4.3, 3. , 1.1, 0.1],
           [5.8, 4. , 1.2, 0.2],
           [5.7, 4.4, 1.5, 0.4],
           [5.4, 3.9, 1.3, 0.4],
           [5.1, 3.5, 1.4, 0.3],
           [5.7, 3.8, 1.7, 0.3],
           [5.1, 3.8, 1.5, 0.3],
           [5.4, 3.4, 1.7, 0.2],
           [5.1, 3.7, 1.5, 0.4],
           [4.6, 3.6, 1. , 0.2],
           [5.1, 3.3, 1.7, 0.5],
           [4.8, 3.4, 1.9, 0.2],
           [5. , 3. , 1.6, 0.2],
           [5. , 3.4, 1.6, 0.4],
           [5.2, 3.5, 1.5, 0.2],
           [5.2, 3.4, 1.4, 0.2],
           [4.7, 3.2, 1.6, 0.2],
           [4.8, 3.1, 1.6, 0.2],
           [5.4, 3.4, 1.5, 0.4],
           [5.2, 4.1, 1.5, 0.1],
           [5.5, 4.2, 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.2],
           [5. , 3.2, 1.2, 0.2],
           [5.5, 3.5, 1.3, 0.2],
           [4.9, 3.6, 1.4, 0.1],
           [4.4, 3. , 1.3, 0.2],
           [5.1, 3.4, 1.5, 0.2],
           [5. , 3.5, 1.3, 0.3],
           [4.5, 2.3, 1.3, 0.3],
           [4.4, 3.2, 1.3, 0.2],
           [5. , 3.5, 1.6, 0.6],
           [5.1, 3.8, 1.9, 0.4],
           [4.8, 3. , 1.4, 0.3],
           [5.1, 3.8, 1.6, 0.2],
           [4.6, 3.2, 1.4, 0.2],
           [5.3, 3.7, 1.5, 0.2],
           [5. , 3.3, 1.4, 0.2],
           [7. , 3.2, 4.7, 1.4],
           [6.4, 3.2, 4.5, 1.5],
           [6.9, 3.1, 4.9, 1.5],
           [5.5, 2.3, 4. , 1.3],
           [6.5, 2.8, 4.6, 1.5],
           [5.7, 2.8, 4.5, 1.3],
           [6.3, 3.3, 4.7, 1.6],
           [4.9, 2.4, 3.3, 1. ],
           [6.6, 2.9, 4.6, 1.3],
           [5.2, 2.7, 3.9, 1.4],
           [5. , 2. , 3.5, 1. ],
           [5.9, 3. , 4.2, 1.5],
           [6. , 2.2, 4. , 1. ],
           [6.1, 2.9, 4.7, 1.4],
           [5.6, 2.9, 3.6, 1.3],
           [6.7, 3.1, 4.4, 1.4],
           [5.6, 3. , 4.5, 1.5],
           [5.8, 2.7, 4.1, 1. ],
           [6.2, 2.2, 4.5, 1.5],
           [5.6, 2.5, 3.9, 1.1],
           [5.9, 3.2, 4.8, 1.8],
           [6.1, 2.8, 4. , 1.3],
           [6.3, 2.5, 4.9, 1.5],
           [6.1, 2.8, 4.7, 1.2],
           [6.4, 2.9, 4.3, 1.3],
           [6.6, 3. , 4.4, 1.4],
           [6.8, 2.8, 4.8, 1.4],
           [6.7, 3. , 5. , 1.7],
           [6. , 2.9, 4.5, 1.5],
           [5.7, 2.6, 3.5, 1. ],
           [5.5, 2.4, 3.8, 1.1],
           [5.5, 2.4, 3.7, 1. ],
           [5.8, 2.7, 3.9, 1.2],
           [6. , 2.7, 5.1, 1.6],
           [5.4, 3. , 4.5, 1.5],
           [6. , 3.4, 4.5, 1.6],
           [6.7, 3.1, 4.7, 1.5],
           [6.3, 2.3, 4.4, 1.3],
           [5.6, 3. , 4.1, 1.3],
           [5.5, 2.5, 4. , 1.3],
           [5.5, 2.6, 4.4, 1.2],
           [6.1, 3. , 4.6, 1.4],
           [5.8, 2.6, 4. , 1.2],
           [5. , 2.3, 3.3, 1. ],
           [5.6, 2.7, 4.2, 1.3],
           [5.7, 3. , 4.2, 1.2],
           [5.7, 2.9, 4.2, 1.3],
           [6.2, 2.9, 4.3, 1.3],
           [5.1, 2.5, 3. , 1.1],
           [5.7, 2.8, 4.1, 1.3],
           [6.3, 3.3, 6. , 2.5],
           [5.8, 2.7, 5.1, 1.9],
           [7.1, 3. , 5.9, 2.1],
           [6.3, 2.9, 5.6, 1.8],
           [6.5, 3. , 5.8, 2.2],
           [7.6, 3. , 6.6, 2.1],
           [4.9, 2.5, 4.5, 1.7],
           [7.3, 2.9, 6.3, 1.8],
           [6.7, 2.5, 5.8, 1.8],
           [7.2, 3.6, 6.1, 2.5],
           [6.5, 3.2, 5.1, 2. ],
           [6.4, 2.7, 5.3, 1.9],
           [6.8, 3. , 5.5, 2.1],
           [5.7, 2.5, 5. , 2. ],
           [5.8, 2.8, 5.1, 2.4],
           [6.4, 3.2, 5.3, 2.3],
           [6.5, 3. , 5.5, 1.8],
           [7.7, 3.8, 6.7, 2.2],
           [7.7, 2.6, 6.9, 2.3],
           [6. , 2.2, 5. , 1.5],
           [6.9, 3.2, 5.7, 2.3],
           [5.6, 2.8, 4.9, 2. ],
           [7.7, 2.8, 6.7, 2. ],
           [6.3, 2.7, 4.9, 1.8],
           [6.7, 3.3, 5.7, 2.1],
           [7.2, 3.2, 6. , 1.8],
           [6.2, 2.8, 4.8, 1.8],
           [6.1, 3. , 4.9, 1.8],
           [6.4, 2.8, 5.6, 2.1],
           [7.2, 3. , 5.8, 1.6],
           [7.4, 2.8, 6.1, 1.9],
           [7.9, 3.8, 6.4, 2. ],
           [6.4, 2.8, 5.6, 2.2],
           [6.3, 2.8, 5.1, 1.5],
           [6.1, 2.6, 5.6, 1.4],
           [7.7, 3. , 6.1, 2.3],
           [6.3, 3.4, 5.6, 2.4],
           [6.4, 3.1, 5.5, 1.8],
           [6. , 3. , 4.8, 1.8],
           [6.9, 3.1, 5.4, 2.1],
           [6.7, 3.1, 5.6, 2.4],
           [6.9, 3.1, 5.1, 2.3],
           [5.8, 2.7, 5.1, 1.9],
           [6.8, 3.2, 5.9, 2.3],
           [6.7, 3.3, 5.7, 2.5],
           [6.7, 3. , 5.2, 2.3],
           [6.3, 2.5, 5. , 1.9],
           [6.5, 3. , 5.2, 2. ],
           [6.2, 3.4, 5.4, 2.3],
           [5.9, 3. , 5.1, 1.8]])




```python

# Find the average iris
avg_iris = np.mean(iris_vectors, axis=0)
avg_iris
```




    array([5.84333333, 3.05733333, 3.758     , 1.19933333])



# Matrices


```python
import numpy as np

N = np.array([[1, 2],[3, 4],[5, 6]])
print(N)
```

    [[1 2]
     [3 4]
     [5 6]]



```python
import numpy as np

N = np.array([[1, 2],[3, 4],[5, 6]])

# The entire first column
n_i_1 = N[:, 0]

n_i_1
```




    array([1, 3, 5])




```python
# The entire first row
n_1_j = N[0,:]
n_1_j
```




    array([1, 2])




```python

# The top left value
n_1_1 = N[0, 0]

n_1_1 
```




    1




```python
# The bottom right value
n_3_2 = N[2, 1]
n_3_2
```




    6



## Matrix Addition and Scalar multiplication


```python
import numpy as np

N = np.array([[1, 2],[3, 4],[5, 6]])

print(N)
```

    [[1 2]
     [3 4]
     [5 6]]



```python
# Scalar multiplication
print(3 * N)

```

    [[ 3  6]
     [ 9 12]
     [15 18]]



```python

# Addition
M = np.arange(7,13).reshape(3, 2)
print(M)
```

    [[ 7  8]
     [ 9 10]
     [11 12]]



```python
print(N + M)

```

    [[ 8 10]
     [12 14]
     [16 18]]



```python

```

## Transpose


```python
import numpy as np

A = np.array([[2, 7, 11],[1, 6, 45]])

A_t = A.transpose()
print(A, '\n\n', A_t)
```

    [[ 2  7 11]
     [ 1  6 45]] 
    
     [[ 2  1]
     [ 7  6]
     [11 45]]


## Transposing vectors

 There are three possible ways to think about vectors in the context of a matrix (or a DataFrame): 
 
 * as row vectors, 
 * as column vectors, 
 * or as neither. 
 
 The Galvanize DSI program will take the position that as a default, __vectors are row vectors__. 
 Then __vT is a column vector__ . However, this is primarily a notational convenience and the other ways of thinking are equally valid.

## Finding the product of two matrices


```python
# As you can see, AB≠BA

# Now, use numpy to complete the same operation:

import numpy as np

matrix_A = np.array([[3, 5], [1, 7]])
matrix_B = np.array([[2, 11],[1, 4]])

A_times_B = np.dot(matrix_A, matrix_B)
B_times_A = np.dot(matrix_B, matrix_A)

print(f'AB =\n {A_times_B}\n')
print(f'BA =\n {B_times_A}')


```

    AB =
     [[11 53]
     [ 9 39]]
    
    BA =
     [[17 87]
     [ 7 33]]


Alternatively, you can use **np.matmul()** with the same parameters - this is specifically for multiplying two-dimensional matrices, 

however the **np.dot()** method is built for both vectors and matrices so it's often easiest to just use that one.

## The dot product

Another very important concept dealing with products in linear algebra is the dot product. The dot product is the result of calculating a⋅b=abT where both a and b are row vectors, therefore bT is a column vector. If a and b are the same length, then you are multiplying a (1×n) row vector with a (n×1) column vector. Using the rules for matrix multiplication above, the result will have shape (1×1)

, which is a scalar (sort of, see note below).

Unlike matrices, where most common multiplication notations are acceptable, finding the dot product specifically calls for the use of the ⋅
operator. There is another operation called the cross product that uses the × operator. That is, v⃗ ⋅w⃗ ≠v⃗ ×w⃗ 

.



```python

```


```python
import numpy as np

v = np.arange(2, 9, 2)
w = np.array([1, 3, 4, 9])

v_dot_w = np.dot(v, w)

# Alternatively call the method using a vector as a calling object
w_dot_v = v.dot(w)

print(f'v * w = {v_dot_w}')
print(f'w * v = {w_dot_v}')

```

    v * w = 110
    w * v = 110


https://blog.naver.com/destiny9720/221407625806

벡터의 내적(Dot Product)

이번 글의 제목은 벡터의 내적인데 이의 영문 표현을 dot product로 표시하였다. 사실 수학에서 벡터의 내적을 뜻하는 단어는 inner product다. 그리고 dot product라는 용어는 한글로 점곱이라고도 번역한다. 하지만 게임 실무에서 점곱이라는 단어는 안 쓴다는 것이 문제다.

사실 수학에서 사용하는 내적의 뜻인 inner product는 추상적인 개념이다. 어떤 두 개의 벡터 쌍이 있고 그 쌍으로부터 만들어지는 스칼라 값이 존재하는데, 이 스칼라 값을 추가로 사용해 생성된 벡터 공간을 내적 공간(inner product space)이라고 한다는 것이다.

앞서 설명했지만 이러한 수학의 정의는 우리가 시각적으로 이해할 수 있는 3차원 세계를 고려하지 않고 임의의 상황에서도 적용가능한 일반적인 성질을 기술한 것이라 머리속으로 이해하기 어렵다.

내적을 이렇게 포괄적인 개념으로 다루면 머리가 아프므로, 우리가 목표로하는 컴퓨터 그래픽에 한정지어 가볍게 살펴보는 것이 필요하다. 통상 x,y,z라 불리는 기저벡터들이 직교하는 3차원의 공간인 유클리드 공간에 존재하는 두 벡터의 내적 연산을 살펴보면 시각적으로 이해할 수 있다. 이러한 유클리드 공간에서의 내적을 다른 말로 스칼라곱(scalar product) 또는 점곱(dot product)이라고 한다.

컴퓨터 그래픽스는 시각적으로 인지하는 3차원 공간에서의 연산을 다루기 때문에 내적을 다룰 때 유클리드 공간의 내적만 고민하면 된다. 그래서 점곱을 사용하고 영문권에서는 dot product라는 용어를 사용한다. 하지만 대한민국의 실무에서는 이를 점곱이라고 하지 않고 내적으로 통칭해서 사용한다. 그래서 제목과 영문 표기가 불일치하는 문제가 발생하는 것이다.

번역에 대한 내용은 이쯤에서 정리하고, 이번에 다룰 주제인 내적은 무언가를 풀기 위해 고안된 공식이 아니고 그냥 두 벡터의 관계를 증명해주는 유용한 계산 방식에 가깝다고 할 수 있다. 

두 벡터 v와 u가 있을 때 내적(dot product)는 다음과 같다. 내적의 중요한 특징 중 하나는 두 벡터를 내적한 결과는 스칼라 값이 된다는 것이다.


이러한 공식으로 인해 같은 벡터의 내적은 벡터의 크기를 제곱한 값이 된다.  같은 벡터를 내적하는 코드는 셰이더 프로그래밍에서 종종 등장하니 알아두면 좋다.


내적을 행렬로 표현하면 다음과 같다. 벡터를 나타내는 행렬과 이의 전치행렬 간의 곱은 내적이 된다.


이렇게 행렬로 내적을 표현하는 방식은 inner product 개념에서 사용하는 공식이다.  하지만 내적을 구하기 위해 ac + bd만 계산하는 것과 두 개의 행렬을 생성하고 하나는 전치 행렬로 변환한 후 행렬과 행렬간의 곱으로 구현하는 것을 비교하면 후자는 무의미한 일이라 할 수 있다. 그래서 우리 게임 수학에서는 inner product를 쓰지 않고 dot product를 쓰는 것이다. 

피타고라스 정리를 사용하면 내적에서 아래와 같은 유의미한 공식이 도출된다.


즉 두 벡터가 직각을 이루면 내적 값은 0이 된다는 것이다.

이를 기반으로 임의의 사이각을 가지는 두 벡터의 내적은 다음과 같은 공식으로 최종 유도된다.  유도 방법에 대한 내용은 우리의 주제를 벗어나니 생략하겠다. 


그런데 잠깐 이는 어디선가 많이 본 듯 한 공식이지 않은가? 삼각형의 세 변을 a, b, c라고 했을 때 지난 글에서 다음과 같은 코사인 법칙이 있었다고 했다. ||a||를 a로 변환해 적용하면 위 공식과 동일하다는 것을 알 수 있다. 


두 벡터의 내적 값이 가지는 속성은 여러가지로 쓸모가 많다.  몇 가지 유용한 사례에 대해 알아보자.


[1. 직교성 판별]  

내적은 두 벡터가 직교인지 아닌지 판단하는데 사용할 수 있다. 두 벡터의 내적 값이 0이면 두 벡터는 직교한다. 앞서서 유도한 회전 행렬을 다시 살펴보자


첫 번째 행벡터인 기저벡터 (cosθ, sinθ)와 두 번째 행벡터 기저벡터 (-sinθ, cosθ)의 내적을 구하면 -cosθsinθ + sinθcosθ이 되어 결과는 0이 됨을 알 수 있다. 열 벡터의 내적 계산도 0이 나온다. 이로써 회전 행렬은 직교 행렬임을 확인할 수 있다. 

앞서서 직교 행렬의 역행렬은 전치행렬이라고 하였다.  이 것도 증명하면 좋지만 인터넷에 많이 나와있으니 생략하겠다.  따라서 회전 행렬의 역행렬은 다음과 같다.


코사인 함수 특성상 cosθ와 cos(-θ) 값은 동일하지만 sin(-θ)는 부호가 달라지는 -sinθ이 된다. 따라서 회전의 역행렬은 -θ만큼 거꾸로 회전시킨 회전 행렬과 동일하다는 것을 알 수 있다.


[2. cos 값 측정]  

내적의 두 번째 유용한 성질은 cosθ 값을 몰라도 곱셈과 덧셈으로 이를 구할 수 있다는 것이다. 이 성질은 특히 벡터의 크기가 1일때 유용하다. 벡터의 크기가 1이면 a와 b의 내적은 바로 cosθ가 되기 때문이다.  내적과 밀접한 관련이 있는 cos 함수의 그래프는 다음과 같다. 

[출처 : https://www.onlinemathlearning.com/trig-graphs.html ]


크기가 1인 두 벡터가 있다면 곱셈과 덧셈을 사용해 cosθ 값을 바로 구할 수 있는데, 컴퓨터 그래픽에서 이를 사용하는 대표적인 공식이 램버트 코사인 법칙(Lambert’s cosine law)이다.

램버트 코사인 법칙은 표면에 빛이 들어오는 각도에 따라 반사되는 세기는 cos 함수에 비례한다는 단순한 법칙이다.


[출처 : http://www.dfisica.ubi.pt/~hgil/Fotometria/HandBook/ch06.html ]

이의 값을 구할 때 코사인 함수 값을 구하지 않고 표면이 향하는 노멀 벡터(N)과 빛으로 향하는 벡터(L)를 구한 후 두 벡터의 내적을 사용하면 컴퓨터로 빠르게 계산할 수 있다.  이를 사용하는 셰이딩 모델을 Diffuse Shading 모델이라고 하며 이를 사용해 명암을 표현한 결과는 아래와 같다.  Diffuse Shading 모델은 컴퓨터 그래픽에서 명암을 표현하는 가장 빠르고 기본적인 모델이며 통칭해서 N dot L 이라고도 한다. 




[3. 시야각 판별]

내적의 세 번째 유용한 성질은 이를 사용해 물체가 앞인지 뒤인지를 빠르게 파악할 수 있다는 것이다. 이 공식은 게임 로직에서 시야 혹은 방향 판별을 위해 많이 사용된다.  내적의 공식을 살펴보면 벡터의 크기 ||a||와 ||b||는 0보다 작을 수 없기 때문에 내적 값의 최종 부호는 결국 cosθ의 값에 의존하게 됨을 알 수 있다. 

θ 값이 -90보다 크고 90보다 작으면 벡터 크기에 관계없이 이 값은 반드시 + 부호를 가지게 되고 90보다 크고 270도보다 작을 때는 - 값을 가지게 된다. 따라서 내적의 값이 양수면 두 벡터는 같은 방향을 향하고 있고 음수면 서로 다른 방향을 향하고 있다는 것을 의미한다. 

이렇게 앞에 있는지 뒤에 있는지를 판별하는 문제의 범위를 확장해보자.  해당 문제의 본질은 전방의 시야각이 180( 왼쪽 90도, 오른쪽 90도 )인 경우라고 할 수 있다. 

그래서 조금 더 응용한다면 아래 그림과 같이 임의의 시야각이 주어졌을 때 ( 회색 영역 ) 시선 벡터와 목표물로 향하는 두 벡터의 내적 값을 계산하고 이 값이 cos(시야각/2) 값보다 크면 목표물은 시야 범위 내에 있는 것으로 판단할 수 있다.  이 때 두 벡터의 크기는 1을 만들어주어야 한다. 



[4. 투영 벡터]

마지막으로 내적을 사용해 하나의 벡터를 다른 벡터에 투영한 벡터를 구할 수 있다. 이 공식은 다양한 분야에서 활용된다. 

  

투영 벡터를 구하는 공식은 세 단계로 나누어진다. 

1. 투영 벡터의 크기를 구하고 결과를 내적의 형태로 변경한다.  ||?||

2. b벡터를 크기가 1인 단위 벡터를 만든다.  b/||b||

3. 1번과 2번을 곱하면 ? 벡터를 구할 수 있다.  

 

위의 공식에서 굳이 코사인으로 구할 수 있는 값을 내적으로 바꾸는 이유는 2번과 동일하다.  

컴퓨터에서 사인 이나 코사인 값을 직접 계산하는 것은 큰 연산 비용을 요구한다.  하지만 내적을 사용하면 컴퓨터가 가장 빠르고 잘 할 수 있는 곱셈과 덧셈의 연산 문제로 귀결된다. 따라서 코사인 함수가 나오는 공식은 모두 내적으로 변경하는 것이 바람직하다. 

[출처] 13. 벡터의 내적(Dot Product)|작성자 이득우

# Identity matrix

An important matrix, related to matrix multiplication, is the identity matrix. The identity matrix is a family of matrices that all share the same properties:

    They are square, same number of rows and columns
    They are diagonal, only non-zero entries have the same row and column index
    All non-zero entries are 1

Notationally, the capital letter I
, with a subscript indicating the number of rows and columns are used to represent Indentity matrices. These matrices serve as the identity element for matrix multiplication. If you have a matrix A with shape (m×n)

, then:

ImA=AIn=A

Here In

means the identity matrix with size n (it has n rows and n columns, because all identity matrices are square). So for any matrix, multiplying by the identity matrix of appropriate size leaves the matrix it is multiplied by unchanged. This may not seem important, but the identity matrix is a very useful concept to use as you develop more complex linear algebra use cases in the future.



# Linear Algebra 2


### Matrix Inversion

####  Verifying matrix inverses  역행렬

In practice, calculating the inverse of a matrix of any reasonable size is done with a computer. Hence, any mathematical computing library will have methods for this. In Python/NumPy, this functionality can be accessed with **np.linalg.inv**.


```python
from numpy.linalg import inv


A = np.array([[ 8,  1, -7], [-2, -8,  5], [-5, -5, -8]])
b = np.array([[-71], [ 25], [-51]])
print(A, '\n\n', b)
```

    [[ 8  1 -7]
     [-2 -8  5]
     [-5 -5 -8]] 
    
     [[-71]
     [ 25]
     [-51]]



```python
A_1 = inv(A)
A_1 
```




    array([[ 0.10102157,  0.04880817, -0.05788876],
           [-0.04653802, -0.1123723 , -0.02951192],
           [-0.03405221,  0.03972758, -0.07037457]])




```python
x = np.linalg.solve(A, b)
x
```




    array([[-3.],
           [ 2.],
           [ 7.]])




#### Systems of Equations Q3


```python
import numpy as np
A = np.array([[ 8,  1, -7], [-2, -8,  5], [-5, -5, -8]])
b = np.array([[-71], [ 25], [-51]])

Ai = np.linalg.inv(A)
Ai


array([[ 0.10102157,  0.04880817, -0.05788876],
       [-0.04653802, -0.1123723 , -0.02951192],
       [-0.03405221,  0.03972758, -0.07037457]])


Ai.dot(b)
array([[-3.],
       [ 2.],
       [ 7.]])
```

### Vector Similarity

컴퓨터가 두 데이터(이미지 혹은 자연어)의 유사성을 측정하는 방법: 유클리드 거리, 코사인 유사도


*데이터의 표현 
컴퓨터는 처리하고자 하는 데이터를 벡터로 다룬다 
예)이미지 처리 분야 : 이미지를 행렬이나 벡터로 표현
예)자연어 처리 분야 : 문장 혹은 단어를 벡터로 표현 

*두 벡터의 유사성을 알아봐야하는 경우가 많다 
예)이미지검색:  두 이미지가 얼마나 유사한지 측정
예)유사문서 검색 : 유사문서 두문장이 얼마나 유사한지 측정


### 유클리드 거리 (Euclide Distance) L2 거리 

유클리드 거리(직선거리)는 피타고라스의 정리를 이용하여 계산할 수 있다 
이 둘 거리를 구해서 가까운 거리가 비슷하다고 할 수 있다. 

텍스트 또한 벡터 값으로 표현했다면 (1차원 배열) 유클리드 거리 계산이 가능하다 

문서1: 컴퓨터가 좋아요
문서2:컴퓨터가 좋아요 좋아요
문서3: 컴퓨터가 싫어요

Bag of Words(BoW)표현 

    
   |컴퓨터| 좋아요| 싫어요<br> 
문서1|     1|   1|    0|<br>
문서2|     1|   2|    0|<br>  
문서3|     1|   0|    1|<br>

문장간의 거리를 측정할 수 있다. 
현재 문서 2와 문서 1이 가까운걸 알 수 있다 

** 유클리드거리가 효과적이지 않은 경우도 있다 

문서1: 컴퓨터가 좋아요
문서2:컴퓨터가 좋아요 컴퓨터가 좋아요  컴퓨터가 좋아요
문서3: 컴퓨터가 싫어요

Bag of Words(BoW)표현 

   컴퓨터 좋아요 싫어요<br> 
문서 1     1   1    0<br>
문서 2     3   3    0<br>  
문서 3     1   0    1<br>

문서 1과 3이 더 가까워 보이게 된당...  암튼...유클리드 거리를 이용하면 효과적이지 않은 경우도 있다 
문서가 짧은거끼리 긴거끼리 유사도가 나오기도 한다 

유클리드 거리는 짧을 수록 유사도가 높다 


d(p⃗ ,q⃗ )=  √ (q1−p1)2+(q2−p2)2+…+(qn−1−pn−1)2+(qn−pn)2

||p⃗ || =  √p21+p22+…+p2n−1+p2n   =  √p⃗ ⋅p⃗ 

d(p⃗ ,q⃗ )  =   ||(q⃗ −p⃗ )||   =  √(q⃗ −p⃗ )⋅(q⃗ −p⃗ )


```python
from numpy.linalg import norm
import numpy as np

def euclidea_distance(A,B):
    return np.linalg.norm(A-B)

document_1 = np.array([1,1,0])
document_2 = np.array([3,3,0])
document_3 = np.array([1,0,1])

print(euclidea_distance(document_1,document_2))
print(euclidea_distance(document_1,document_3))
# 유클리드 거리는 짧을 수록 유사도가 높다 

```

    2.8284271247461903
    1.4142135623730951
    3.7416573867739413


## 코싸인 유사도 
벡터의 크기는 고려하지 않고 두 벡터 사이의 각도만 고려하는 측정법이다
방향이 얼마나 유사한지 1부터 -1까지 사이의 값으로 표현한다. (방향이 동일한 경우1)

1에 가까울수록 서로 유사도가 비슷하다라고 할 수 있다 

![image.png](attachment:image.png)

p⃗ ⋅q⃗ =||p⃗ || ||q⃗ ||cosθ


sim(p⃗ ,q⃗ ) =  cosθ  =   p⃗ ⋅q⃗   /  ||p⃗ || ||q⃗ ||



```python
import numpy as np
from numpy import dot
from numpy.linalg import norm
def cosine_similarity(A,B):
    return dot(A,B) / (norm(A) * norm(B))

document_1 = np.array([1,1,0])
document_2 = np.array([3,3,0])
document_3 = np.array([1,0,1])

print(cosine_similarity(document_1,document_2))
print(cosine_similarity(document_1,document_3))

# 코싸인 유사도는 1에 가까울 수록 유사도가 높다 
```

    1.0
    0.4999999999999999


- 두 벡터간의 유사성을 계산하는 측정법들을 알아보았다 <br>
    - 유클리드 거리, 코싸인 유사도 <br>
- 물론 특정한 데이터 (이미지 문장 오디오등)를 벡터로 표현하는 방법은 별개의 문제 입니다 <br>
   - 데이터에서 특징을 잘 추출하여 벡터로 표현하는 방법에 대한 연구분야가 있습니다 <br>
        - 이미지  처리 분야 - Metric Learning <br>
        - 자연어 처리 분야 -  Word Embedding <br>


```python
from numpy.linalg import norm
import numpy as np

def euclidea_distance(A,B):
    return np.linalg.norm(A-B)

a = np.array([1,1])
b = np.array([-1,0])
c = np.array([4,2])

print(euclidea_distance(a,b))
print(euclidea_distance(a,c))
print(euclidea_distance(b,c))
# 유클리드 거리는 짧을 수록 유사도가 높다 

```

    2.23606797749979
    3.1622776601683795
    5.385164807134504


The distances between these vectors are:

d(a⃗ ,b⃗ )=(−1−1)2+(0−1)2−−−−−−−−−−−−−−−−√=5√≈2.24<br>
d(a⃗ ,c⃗ )=(4−1)2+(2−1)2−−−−−−−−−−−−−−−√=10−−√≈3.16<br>
d(b⃗ ,c⃗ )=(4+1)2+(2−0)2−−−−−−−−−−−−−−−√=29−−√≈5.39<br>


```python
import numpy as np
from numpy import dot
from numpy.linalg import norm
def cosine_similarity(A,B):
    return dot(A,B) / (norm(A) * norm(B))

a = np.array([1,1])
b = np.array([-1,0])
c = np.array([4,2])

print(cosine_similarity(a,b))
print(cosine_similarity(a,c))
print(cosine_similarity(b,c))
# 코싸인 유사도는 1에 가까울 수록 유사도가 높다 
```

    -0.7071067811865475
    0.9486832980505138
    -0.8944271909999159


The Cosine similarity between these vectors are:

sim(b⃗ ,c⃗ )=−1×4+0×2(−1)2+02−−−−−−−−−√42+22−−−−−−√≈−0.89<br>
sim(a⃗ ,b⃗ )=1×−1+1×012+12−−−−−−√(−1)2+02−−−−−−−−−√≈−0.71<br>
sim(a⃗ ,c⃗ )=1×4+1×212+12−−−−−−√42+22−−−−−−√≈0.95<br>
<br>
Keep in mind that the ordering in this question is from least alike to most alike, <br>
while the ordering in the previous question was from most alike to least alike.


```python

```


```python
B = np.array([[ -5,  3, -1], [-3, 6,  -1], [-1, -7, 3]])
Bi = np.linalg.inv(B)
def format_matrix_string(mat):
    """
    Convert a NumPy matrix/array to a string where rows are separated by the
    newline character '\n'.
    INPUT: mat: NumPy array
    OUTPUT: str
    """
    rowstrs = [', '.join(row) for row in np.round(mat, 2).astype(str)]
    return ' \n '.join(rowstrs)

print(format_matrix_string(Bi))
```

    -0.21, 0.04, -0.06 
     -0.19, 0.31, 0.04 
     -0.52, 0.73, 0.4



```python
c = np.array([[-10], [18], [-38]])
Bi.dot(c)
```




    array([[5.],
           [6.],
           [3.]])




```python
p = np.array([8, -3, 8, 5, -5])
q = np.array([-4, 8, 7, -1, 6])

print(euclidea_distance(p,q))
```

    20.566963801203133



```python
p = np.array([8, -3, 8, 5, -5])
q = np.array([-4, 8, 7, -1, 6])

print(cosine_similarity(p,q))
```

    -0.19865211674728586


#### Linear Algebra 1 Checkpoint Q5   ANSWER 


```python
>>> P = np.array([8, -3, 8, 5, -5])
>>> Q = np.array([-4, 8, 7, -1, 6])
>>> distance = np.linalg.norm(P - Q)
>>> distance
20.566963801203133
```




    20.566963801203133



#### Linear Algebra 1 Checkpoint Q6  ANSWER 



```python
>>> P.dot(Q) / (np.linalg.norm(P) * np.linalg.norm(Q))
-0.19865211674728586
```




    -0.19865211674728586



### Check if a Matrix is Invertible


```python

# Function to get cofactor of  mat[p][q] in temp[][]. n is current dimension of mat[][] 
def getCofactor(mat, temp, p, q, n): 
    i = 0
    j = 0
  
    # Looping for each element 
    # of the matrix 
    for row in range(n):  
          
        for col in range(n): 
              
            # Copying into temporary matrix 
            # only those element which are  
            # not in given row and column 
            if (row != p and col != q) : 
                  
                temp[i][j] = mat[row][col] 
                j += 1
  
                # Row is filled, so increase  
                # row index and reset col index 
                if (j == n - 1): 
                    j = 0
                    i += 1
  
# Recursive function for  
# finding determinant of matrix. 
# n is current dimension of mat[][].  
def determinantOfMatrix(mat, n): 
    D = 0 # Initialize result 
  
    # Base case : if matrix  
    # contains single element 
    if (n == 1): 
        return mat[0][0] 
          
    # To store cofactors 
    temp = [[0 for x in range(N)]  
               for y in range(N)]  
  
    sign = 1 # To store sign multiplier 
  
    # Iterate for each  
    # element of first row 
    for f in range(n): 
          
        # Getting Cofactor of mat[0][f] 
        getCofactor(mat, temp, 0, f, n) 
        D += (sign * mat[0][f] *
              determinantOfMatrix(temp, n - 1)) 
  
        # terms are to be added  
        # with alternate sign 
        sign = -sign 
    return D 
  
def isInvertible(mat, n): 
    if (determinantOfMatrix(mat, N) != 0): 
        return True
    else: 
        return False
      
# Driver Code 
mat = [[ 1, 0, 2, -1 ], 
       [ 3, 0, 0, 5 ], 
       [ 2, 1, 4, -3 ], 
       [ 1, 0, 5, 0 ]]; 
      
N = 4
if (isInvertible(mat, N)): 
    print("Yes") 
else: 
    print("No") 
  
# This code is contributed 
# by ChitraNayal 

```


$\color{red}{\text{In order to be invertible, a matrix must be square and full-rank. }}$

# Linear Algebra 3

Transform data using methods from linear algebra including: rotation, projection, and eigen-decomposition


```python

```

### Eigenvectors and Eigenvalues


```python
v = np.array([[-2], [0], [1]])
a = np.array([[5, 8, 16], [4, 1, 8], [-4, -4, -11]])
```


```python

import numpy as np 
  
a = np.array([[5, 8, 16], [4, 1, 8], [-4, -4, -11]])  
v = np.array([[-2], [0], [1]])
  
# Original matrix 
print(a) 
print("") 
print(v) 
print("") 
A, V = np.linalg.eig(a) 
  
# Eigenvalues of the said matrix" 
print(A) 
print("") 
  
# Eigenvectors of the said matrix 
print(V) 

```

    [[  5   8  16]
     [  4   1   8]
     [ -4  -4 -11]]
    
    [[-2]
     [ 0]
     [ 1]]
    
    [ 1. -3. -3.]
    
    [[-0.81649658 -0.38329359 -0.91170492]
     [-0.40824829  0.88842382  0.13714837]
     [ 0.40824829 -0.25256511  0.38727828]]


![image.png](attachment:image.png)


```python

```

### Comprehension Challenges
Linear Algebra 3 Checkpoint Q2


```python
>>> import numpy as np
>>> a = np.array([[3,1], [0,2]])  # establish a
>>> a.dot(np.array([1,0]))  # check v1
array([3, 0])  # looks like 3 times v1, so v1 is in


>>> a.dot(np.array([np.sqrt(3)/2, 0.5]))  # check v2
array([3.09807621, 1.        ])  # ok, so 1 is clearly twice 0.5, so is 3.098 twice sqrt(3)/2?
>>> np.sqrt(3)
1.7320508075688772  # guess not! v2 is out


>>> a.dot(np.array([-np.sqrt(2)/2, np.sqrt(2)/2]))  # check v3
array([-1.41421356,  1.41421356])  
# still symmetric values where the first one is negative, 
# so there exists some scalar for this vector that works as lambda, so this is an eigenvector; v3 is in
#  So, v1 and v3 are in, v2 is out.
```


```python
a = np.array([[3, 1], [0, 2]])

v1 = np.array([[1], [0]])
v2 = np.array([np.sqrt(3)/2, 0.5]) 
v3 =np.array([-np.sqrt(2)/2, np.sqrt(2)/2])

w, v = np.linalg.eig(a) 
print(w)
print(v)
```

    [3. 2.]
    [[ 1.         -0.70710678]
     [ 0.          0.70710678]]



```python

```


```python

```


```python

```
