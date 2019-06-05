
dict ={}
list = []
with open('data','r') as f:
    while 1 :
        number = f.readline()
        list.append(number)
        if not number:
            break


for number in list:
    key = number[0:6]
    if key in dict:
        dict[key].append(number)
    else:
        list = []
        list.append(number)
        dict[key] = list


for k in dict.keys():
    if len(dict[k]) == 10 :
        dict[k] = k

kl = []
with open('rs.txt','w') as f:
    for k in dict.keys():
        if dict[k].__class__ == str:
            print(dict[k])
            f.write(dict[k]+'\n')
        else :
            for v in dict[k]:
                f.write(v)
                print(v)









