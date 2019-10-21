import sys

IDX=2
li = [([0] * IDX) for i in range(IDX)]
ground_truth = ([0] * IDX)
predict = ([0] * IDX)

def load_dict(filename):
    cdict = {}
    for line in open(filename):
        line = line.strip()

        sep = line.split("\t")

        id = int(sep[0])
        label = (sep[1])

        cdict[label] = id


    return cdict

cdict = load_dict("id.to.cat")

N = 0
for line in open(sys.argv[1]):
    line = line.strip()
    sep = line.split("\t")

    label = (sep[1])
    pre = (sep[2])
    list_sep = sep[3].split()

    if label == (sys.argv[3]) :
        label = 0
    else:
        label = 1

    id = cdict.get(sys.argv[3])
    score = float(list_sep[id]) 

    if score >= float(sys.argv[2]) :
        pre = 0
    else:
        pre = 1

    li[pre][label] += 1
    N += 1
    ground_truth[label] += 1
    predict[pre] += 1

cnt = 0
all_acc = 0
all_recall = 0
all_f1 = 0
for i in range(1):
    right = li[i][i]
    wrong = predict[i] - right
    all = ground_truth[i]

    cnt += 1
    if all == 0 or (right + wrong) == 0:
        print (i, 0, 0, 0)
    else:
        acc = right * 1.0/ (right + wrong)
        recall = right * 1.0 / all

        if acc + recall == 0:
            f1 = 0
        else:
            f1 = 2 * acc * recall/(acc + recall)

        all_acc += acc
        all_recall += recall
        all_f1 += f1
        print (i, round(acc,3), round(recall,3), round(f1,3))

for each_li in li:
    each_sum = sum(each_li)
    if each_sum == 0 :
        new_li = [str(i) for i in each_li]
    else:
        new_li = [str(i) for i in each_li]

    new_line = "\t".join(new_li)
    print (new_line)
