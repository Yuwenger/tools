#coding:utf-8

#画二分类的p-r曲线
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#只取两类用以二分类
# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)

# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
#得到分数，这个分数跟神经网络不同，只是一维的，它划分两类的依据是分数的大小阈值。其实神经网络也是一样的，因为
#两类的分数之和为1
y_score = classifier.decision_function(X_test)


from sklearn.metrics import average_precision_score
#计算平均精度
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
#获取召回率和精度
precision, recall, _ = precision_recall_curve(y_test, y_score,pos_label=0)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post',label= 'step')
plt.plot(recall,precision,label= 'plot')
#plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend()
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()