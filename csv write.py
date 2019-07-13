import os
import csv

with open('test.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name','label'])
    i = 0
    with open('test1.csv', 'r', newline='') as f1:
        for each_line in f1:
            print(each_line)
            writer.writerow([each_line[:12], i])
            i = i + 1