def calculateRatio(AX1, AY1, AX2, AY2, BX1, BY1, BX2, BY2):
    SA = abs(AX2 - AX1) * abs(AY2 - AY1)
    SB = abs(BX2 - BX1) * abs(BY2 - BY1)
    SI = max(0, min(AX2, BX2) - max(AX1, BX1)) * \
        max(0, min(AY2, BY2) - max(AY1, BY1))
    SU = SA + SB - SI
    ratio = float(SI / float(SU))
    return ratio


def get_accuracy(boxes1, boxes2):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    accuracy = 0.0
    precision = 0.0
    recal = 0.0
    for i in range(len(boxes1)):
        (match1, AX1, AY1, AX2, AY2) = boxes1[i]
        (match2, BX1, BY1, BX2, BY2) = boxes2[i]
        if(match1 and match2):
            ratio = calculateRatio(AX1, AY1, AX2, AY2, BX1, BY1, BX2, BY2)
            print(i, ratio)
            if ratio > .5:
                TP += 1
            else:
                FP += 1
        elif(match2 == True and match1 == False):
            FP += 1

        elif(match2 == False and match1 == True):
            FN += 1

        elif(match2 == False and match1 == False):
            TN += 1
    if TP + FP != 0:
        precision = TP / (TP + FP)

    if TP + FN != 0:
        recal = TP / (TP + FN)

    if TP + TN + FP + FN != 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("TP ", TP, " TN ", TN, " FP ", FP, " FN ", FN)
    print("Accuracy = %f , precision = %f , recal = %f" %
          (accuracy, precision, recal))


if __name__ == '__main__':
    f1 = open('output.txt', 'r')
    f2 = open('SURFRectangles.txt', 'r')
    rectanglesFD = []
    rectanglesre = []
    for i in range(600):
        [j, matched1, x, y, x2, y2] = map(int, f1.readline().split())
        rectanglesFD.append((True, x, y, x2, y2))
        [j, matched2, x, y, x2, y2] = f2.readline().split()
        j = int(j)
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)
        rectanglesre.append((matched, x, y, x2, y2))
    get_accuracy(rectanglesFD, rectanglesre)
