import os
import re
import math
#读取总体文本
def novel_get(path):
    novel_contents = ''
    names = os.listdir(path)
    for name in names:
        novel_path = path + '/' + name
        f = open(novel_path, 'r', encoding='gbk', errors='ignore')
        novel_contents += f.read()
        f.close()
    return novel_contents
#读取单个文本
def onenovel_get(path):
    f=open(path,'r', encoding='gbk', errors='ignore')
    data=f.read()
    return data
#去除无用信息和非中文字符
def novel_delete(data):
    data = data.replace('本书来自www.cr173.com免费txt小说下载站','')
    data = data.replace('更多更新免费电子书请关注www.cr173.com','')
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', data)
    return chinese

#计算一元词频并排序
def unit_fre(data):
    unit = {}
    for word in data:
        unit[word] = unit.get(word, 0) + 1
    newunit = dict(sorted(unit.items(), key=lambda x: x[1], reverse=True))
    return newunit
#计算一元信息熵
def unit_entropy(data):
    entropy = 0
    total = sum(data.values())
    for word in data.keys():
        p_x = data[word] / total
        entropy -= p_x * math.log2(p_x)
    return entropy

#计算二元词频并排序
def double_fre(data):
    double={}
    for i in range(len(data)-1):
        double[(data[i],data[i+1])]=double.get((data[i], data[i + 1]), 0) + 1
    newdouble = dict(sorted(double.items(), key=lambda x: x[1], reverse=True))
    return newdouble

#计算二元信息熵
def double_entropy(data,unitfre):
    entropy=0
    total=sum(data.values())
    for key1, key2 in data.keys():
        p_x = data[key1, key2] / total
        p_xy = data[key1, key2] / unitfre[key1]
        entropy -= p_x * math.log2(p_xy)
    return entropy

#计算三元词频并排序
def tri_fre(data):
    tri = {}
    for i in range(len(data) - 2):
        tri[(data[i], data[i + 1], data[i + 2])] = tri.get((data[i], data[i + 1], data[i + 2]), 0) + 1
    newtri = dict(sorted(tri.items(), key=lambda x: x[1], reverse=True))
    return newtri
#计算三元信息熵
def tri_entropy(data,doublefre):
    entropy=0
    total = sum(data.values())
    for key1, key2, key3 in data.keys():
        p_x = data[key1, key2, key3] / total
        p_xyz = data[key1, key2, key3] / doublefre[key1, key2]
        entropy -= p_x * math.log2(p_xyz)
    return entropy



if __name__ == '__main__':
    print('------------------------------------------')
    path='C:/Users/zzy/Desktop/课程相关/001_NLP/文本'
    file=os.listdir(path)
    for txt in file:
        position=path+'/'+txt
        onenovel=onenovel_get(position)
        onenovel = novel_delete(onenovel)
        ufre = unit_fre(onenovel)
        unit = unit_entropy(ufre)
        dfre = double_fre(onenovel)
        double = double_entropy(dfre, ufre)
        tfre = tri_fre(onenovel)
        tri = tri_entropy(tfre, dfre)
        txt=txt.replace('.txt','')
        print(txt+'的一元信息熵为' + str(unit))
        print(txt+'的二元信息熵为' + str(double))
        print(txt+'的二元信息熵为' + str(tri))
        print('-------------------------------------')
    novel=novel_get('C:/Users/zzy/Desktop/课程相关/001_NLP/文本')
    novel=novel_delete(novel)
    ufre=unit_fre(novel)
    unit=unit_entropy(ufre)
    dfre=double_fre(novel)
    double=double_entropy(dfre,ufre)
    tfre=tri_fre(novel)
    tri=tri_entropy(tfre,dfre)
    print('一元信息熵为'+str(unit))
    print('二元信息熵为' + str(double))
    print('二元信息熵为' + str(tri))