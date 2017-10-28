# coding=utf-8
import numpy as np
import xlrd
workbook = xlrd.open_workbook('X1_0721_400_1.xlsx')
print('worksheets is %s' % workbook.sheet_names())

sheet1 = workbook.sheet_by_name(u'Sheet1')
list1_1=[]
for i in range(4000):
    list1_1.append(np.array(sheet1.row_values(i)))
X_test=np.asarray(list1_1)
list1_2=[]
for i in range(4000,40000,1):
    list1_2.append(np.array(sheet1.row_values(i)))
X_train=np.asarray(list1_2)

sheet2 = workbook.sheet_by_name(u'Sheet2')
list2_1=[]
for i in range(4000):
    list2_1.append(np.array(sheet2.row_values(i)))
y_test=np.asarray(list2_1)
list2_2=[]
for i in range(4000,40000,1):
    list2_2.append(np.array(sheet2.row_values(i)))
y_train=np.asarray(list2_2)





