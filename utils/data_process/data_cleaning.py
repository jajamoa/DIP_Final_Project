import os
from photohash import distance
from tqdm import tqdm

def rm_list(dir:str):
    assert os.path.isdir(dir)
    assert os.path.exists(dir)

    rm=[]
    namelist=os.listdir(dir)
    i=0
    while i<len(namelist):
        if not namelist[i].startswith('CT'):
            rm.append(os.path.join(dir,namelist[i]))
            namelist.pop(i)
            i-=1
        i+=1

    for i in range(len(namelist)):
        if i and i!=len(namelist)-1:
            left=os.path.join(dir,namelist[i-1])
            this=os.path.join(dir,namelist[i])
            right=os.path.join(dir,namelist[i+1])
            if distance(left,this)>5 and distance(this,right)>5:
                # print(left)
                # print(this)
                # print(right)
                rm.append(this)
                if i==1:
                    if distance(left,right)>5:
                        rm.append(left)
                elif i==len(namelist)-2:
                    if distance(left,right)>5:
                        rm.append(right)

    return rm

# for row in rm_list(r'D:\工作日志\大三上\DIP\trainData\subject-level\Cap\cap000'):
#     print(row)


def remove(dir:str):
    assert os.path.isdir(dir)
    assert os.path.exists(dir)

    subdir_list=os.listdir(dir)
    for subdir in tqdm(subdir_list):
        rm_root=os.path.join(dir,subdir)
        if os.path.isdir(rm_root):
            rml = rm_list(rm_root)
            for d in rml:
                assert os.path.exists(d)
                os.remove(d)


remove(r'D:\工作日志\大三上\DIP\trainData\subject-level\Cap')
remove(r'D:\工作日志\大三上\DIP\trainData\subject-level\Covid-19')
remove(r'D:\工作日志\大三上\DIP\trainData\subject-level\Non-infected')

