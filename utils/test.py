with open("data/cutclean_label_corpus10000.txt", encoding="utf-8") as f:
    datas = f.readlines()

bd='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． abcdefghijklmnopqrstuvwxyzsABCDEFGHIJKLMNOPQRSTUVWXYZ'
data = set()
data2 = set()
data3 = set()
for line in datas:
        for i in line.split():   
            data3.add(i)   
        for i in bd:  
            line = line.replace(i, '')
        l = line.split()
        for i in l:
            if i != '':
                data.add(i)
            data2.add(i)

print(len(data))
print(len(data2))
print(len(data3))

7765
7765
30065