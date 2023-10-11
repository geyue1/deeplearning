# -* UTF-8 *-
'''
==============================================================
@Project -> File : pytorch -> util.py
@Author : yge
@Date : 2023/3/30 14:51
@Desc :

==============================================================
'''
import os.path


def del_files(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        file_list = os.listdir(path)
        for file_name in file_list:
            file = os.path.join(path,file_name)
            del_files(file)

if __name__ == "__main__":
    del_files("../../board")