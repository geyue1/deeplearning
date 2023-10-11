# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> dp.py
@Author : yge
@Date : 2023/8/10 10:13
@Desc :

==============================================================
'''

price_dict = {1:1,2:5,3:8,4:9,5:0,6:17,7:17,8:20,9:24,10:30}

def cut_rod_buttom_up(n):
    if n==1:
        return 1
    a = []
    for i in range(1,n+1):
        value = -1
        for j in range(1,i+1):
            value = max(value,)
