# coding=utf-8
import numpy as np
import xlrd
workbook = xlrd.open_workbook('alldata.xlsx')
print('worksheets is %s' % workbook.sheet_names())

worksheet1 = workbook.sheet_by_name(u'Sheet1')
list1=[]
for curr_row in range(4000):
    list1.append(np.array(worksheet1.row_values(curr_row)))
test_set=np.asarray(list1)

worksheet2 = workbook.sheet_by_name(u'Sheet2')
list2=[]
for curr_row in range(4000):
    list2.append(np.array(worksheet2.row_values(curr_row),dtype=np.int32))
test_label=np.asarray(list2)

worksheet3 = workbook.sheet_by_name(u'Sheet3')
list3=[]
for curr_row in range(36000):
    list3.append(np.array(worksheet3.row_values(curr_row)))
train_set=np.asarray(list3)

worksheet4 = workbook.sheet_by_name(u'Sheet4')
list4=[]
for curr_row in range(36000):
    list4.append(np.array(worksheet4.row_values(curr_row),dtype=np.int32))
train_label=np.asarray(list4)





