# -*- coding: utf-8 -*-
# @Time    : 2018/3/8 21:32
# @Author  : Yuweng
# @Email   :
# @File    :
# @Function: 多分类p-r曲线绘制

from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# Use label_binarize to be multi-label like settings
#相当于one-hot编码比如，0变为[1,0,0]，1变为[0,1,0],标签2变为[0,0,1]
Y = label_binarize(y, classes=[0, 1, 2])
print y
print 'Y'
print Y.shape
#Y.shape  [num_samples,num_class]
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
#y_score 为[num_samples,num_class]数组
y_score = classifier.decision_function(X_test)
print y_score