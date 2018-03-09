# -*- coding: utf-8 -*-
# @Time    : 2018/3/8 21:32
# @Author  : Yuweng
# @Email   :
# @File    :
# @Function: 多分类p-r曲线绘制和roc曲线绘制

########################
#步骤：可以先求得样本的分数（比如神经网络求得），然后将标签one-hot编码，最后求取召回率和精度
#mAP：  根据论文中可知，mAP就是各个种类的AP的均值（参见yolo9000）
###########################

from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(1)
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
#由此处可见将标签变为one-hot编码不是必须的，因为分类器只需要得到相应的分数即可。即y_score 为[num_samples,num_class]数组
classifier.fit(X_train, Y_train)
#y_score 为[num_samples,num_class]数组
y_score = classifier.decision_function(X_test)
print y_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
#计算每一类的精度和召回率
# For each class
precision = dict()
recall = dict()
average_precision = dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
#此处才真正体现one-hot编码的作用，即将本类体现出来
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])

    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    #获取tpr和fpr画roc曲线
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i],tpr[i])

import matplotlib.pyplot as plt
colors = ['navy', 'turquoise', 'darkorange']

plt.figure(figsize=(7, 8))
for i in range(n_classes):
    plt.plot(recall[i],precision[i],color = colors[i],label = 'class {},AP :{:0.2f}'.format(i,average_precision[i]))
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision/Recall curve')
plt.legend()


plt.figure(figsize=(7, 8))
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],color = colors[i],label = 'class {},auc :{:0.2f}'.format(i,roc_auc[i]))
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.legend()
plt.show()
