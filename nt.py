# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:03:59 2019

@author: Zewditu
"""

from sklearn import preprocessing
# Normalize total_bedrooms column
X_array = np.array(dataset['price'])
normalized_X = preprocessing.normalize([X_array])

# Get column names first
names = dataset.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
scaled_dataset = scaler.fit_transform(dataset)
scaled_dataset = pd.DataFrame(scaled_dataset, columns=names)
