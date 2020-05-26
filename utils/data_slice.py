import numpy as np
import csv

train_len = 34471
test_len = 4000

event = {
    '文体娱乐':0,
    '社会生活':1,
    '科技':2,
    '政治':3,
    '军事':4,
    '财经商业':5,
    '教育考试':6,
    '医药健康':7
}
gender = {
    '男':0,
    '女':1
}

def get_train_index():
    with open("train.txt", "r") as train_file:
        index = train_file.read().split()
        index = [int(i) for i in index]
    return index

def get_test_index():
    with open("test.txt", "r") as test_file:
        index = test_file.read().split()
        index = [int(i) for i in index]
    return index

def test_w2v():
    with open("text.txt", "w", encoding="utf-8") as tw_file:
        with open('data/train.csv', encoding="utf-8")as f:
            data = csv.reader(f)
            for line in data:
                if (len(line[-1]) < 2) and (int(line[-1]) < 2):
                    tw_file.write(line[1])

def load_texts_and_labels(path="data/train.csv"):
    with open(path, encoding="utf-8")as f:
        texts = []
        labels = []
        data = csv.reader(f)
        for line in data:
            if (len(line[-1]) < 2) and (int(line[-1]) < 2):
                texts.append(line[1])
                labels.append(line[-1])
        return texts, labels

def load_all(path="data/train.csv"):
    with open(path, encoding="utf-8")as f:
        texts = []
        pics = []
        events = []
        gd = []
        follow = []
        fans = []
        weibo = []
        labels = []
        data = csv.reader(f)
        #i = 0
        for i,line in enumerate(data):
            #print(i)
            if (len(line[-1]) < 2) and (int(line[-1]) < 2):# and (line[3] != '') and (len(line[4]) != 0) and (len(line[5]) != 0) and (len(line[6]) != 0): #3, 1037, 19271, 19403
                #print(line[1], event[line[9]], line[10])
                #if  ((event[line[9]] == 3) or (event[line[9]] == 4)) or  ((line[3] != '') and (len(line[4]) != 0) and (len(line[5]) != 0) and (len(line[6]) != 0)):
                    texts.append(line[1])
                    pics.append(line[2])
                    '''if (event[line[9]] == 3) and not((line[3] != '') and (len(line[4]) != 0) and (len(line[5]) != 0) and (len(line[6]) != 0)):
                        print(i, event[line[9]])
                    if (event[line[9]] == 4) and not((line[3] != '') and (len(line[4]) != 0) and (len(line[5]) != 0) and (len(line[6]) != 0)):
                        print(i, event[line[9]])'''#36443 36025 2028 1962
                    #gd.append(gender[line[3]])
                    #follow.append(int(line[4]))
                    #fans.append(int(line[5]))
                    #weibo.append(int(line[6]))
                    events.append(event[line[9]])
                    labels.append(int(line[10]))
            '''i = i+1
            if i > 100:
                return texts, labels, events, pics'''
        return texts, labels, np.array(gd), np.array(follow), np.array(fans), np.array(weibo), events, pics

def events_divi(events):
        
    divi = [[] for i in range(8)]
    for i, e in enumerate(events):
        divi[e].append(i)
    divi = np.array(divi)
    return divi

if __name__ == "__main__":
    '''train = open("train.txt", 'w')
    test = open("test.txt", 'w')
    a = (np.arange(38471))
    np.random.shuffle(a)
    a = a.tolist()
    b = [ str(i) for i in a ]
    print(a[9])
    train.write(" ".join(b[0:34471]))
    test.write(" ".join(b[34471:38471]))

    train.close()
    test.close()'''
    #load_all("../data/train.csv")
    texts, labels, gd, follow, fans, weibo, events, pics = load_all("../data/train.csv")
    divi = [[] for i in range(8)]
    for i, e in enumerate(events):
        divi[e].append(i)
    divi = np.array(divi)
    for i in divi:
        print(len(i))
    print("gender")
    for i in divi:
        print(np.mean(gd[np.array(i)]))
    print("follow")
    for i in divi:
        print(np.mean(follow[np.array(i)]))
    print("fans")
    for i in divi:
        print(np.mean(fans[np.array(i)]))
    print("weibo")
    for i in divi:
        print(np.mean(weibo[np.array(i)]))

    #2883 23221 296 1546 416 1601 1008 7025
'''gender
0.4065209850849809
0.40411696309375134
0.2972972972972973
0.2056921086675291
0.21875
0.2679575265459088
0.4126984126984127
0.4990747330960854
follow
640.4477974332293
691.583652728134
782.7364864864865
889.617723156533
689.1826923076923
779.1161773891318
787.6537698412699
616.375231316726
fans
815027.8931668401
1200339.9751948668
744943.4932432432
602009.3473479948
741107.4903846154
1066035.0024984386
3160406.1617063493
822480.95658363
weibo
12049.563302115852
15151.677404073898
15455.462837837838
13969.567917205692
13463.596153846154
15514.791380387258
24783.6626984127
11005.652384341636'''

'''
3325 3
3403 4
4019 3
6548 3
7502 4
8972 3
10563 3
11624 4
12709 3
13757 3
13981 4
17670 3
19342 4
20889 4
20950 4
21195 4
21756 4
21820 4
21896 4
21991 4
22289 4
22407 4
22690 4
22980 4
23140 4
23309 4
23838 3
23955 3
24367 4
26015 4
26587 4
26928 4
27151 4
27620 4
27648 4
28490 4
28586 4
28596 3
28957 4
29144 4
29941 4
30125 4
30326 4
30555 4
30783 4
31254 4
32175 4
32571 4
32781 4
32930 4
33337 4
33613 4
34137 3
34232 4
35120 4
35198 4
35452 4
35493 4
35508 4
35564 4
35599 4
35602 4
35950 4
36737 4
36878 4
38172 4'''
