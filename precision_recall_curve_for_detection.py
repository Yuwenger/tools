# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 11:45
# @Author  : Yuweng
# @Email   : 
# @File    : precision_recall_curve_for_detection.py
# @Function: 求取目标检测的p-r曲线


#首先明确检测出了多少个bbox，然后判断每个bbox的ground truth（通过计算IOU），可以转化为one-hot编码形式，
#这就相当于分类里面的标签，假设有M个bbox，则相当于分类中有M个待分物体，然后每个bbox会有类别的分数，这就转化为
#了分类里面的p-r曲线画法，或者roc画法，只需要带入API即可。