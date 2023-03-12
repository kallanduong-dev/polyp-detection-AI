from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

def avEntropy(arr):
    entropys = []

    for i in range(len(arr)):
        entropys.append(entropy([arr[i],1-arr[i]],base=2))

    return [entropys, statistics.median(entropys)]


def entropyofAverage(arr):
    averageCon = statistics.median(arr)
    return entropy([averageCon,1-averageCon],base=2)


avcon = []

cases = []
for i in range(101):
    cases.append([])

y_true = []
y_scores = []

#some cases had ground truth bounding boxes that were obviously incorrect/inconsistent with each other
#remove inconsistent cases from detections

offAnnotatedCases = [1,2,7,12,29,34,36,37,38,44,51,57,59,61,67,68,72,74,83,97,101]

with open("result.txt") as filestream:
    content = filestream.readlines()
    for i in range(len(content)):
        if content[i].find("case") >= 0:
            caseIndex = int(content[i].find("case"))+4
            if content[i][caseIndex+2].isnumeric(): 
                case = int(content[i][caseIndex:caseIndex+3])
            elif content[i][caseIndex+1].isnumeric():
                case = int(content[i][caseIndex:caseIndex+2])
            else:
                case = int(content[i][caseIndex])

            #ignore inconsistent cases
            if case not in offAnnotatedCases:

                fileIndex = int(content[i].find("file"))+4
                if content[i][fileIndex+2].isnumeric(): 
                    file = int(content[i][fileIndex:fileIndex+3])
                elif content[i][fileIndex+1].isnumeric():

                    file = int(content[i][fileIndex:fileIndex+2])
                else:
                    file = int(content[i][fileIndex])

                #get ground truth bbox 
                with open(f"annotation_txt/case{case}.txt","r") as newfilestream:
                    annotation = newfilestream.readlines()
                    line = annotation[file-1]
                    currentline = line.split(" ")
                    vals = currentline[1].split(",")
                    xmin = int(vals[0])
                    ymin = int(vals[1])
                    xmax = int(vals[2])
                    ymax = int(vals[3])

                #get detection bbox
                if content[i+2].find("polyp: ") >= 0:
                    lxIn = int(content[i+2].find("left_x:"))+9
                    lx = int(content[i+2][lxIn:lxIn+3])

                    tyIn = int(content[i+2].find("top_y:"))+8
                    ty = int(content[i+2][tyIn:tyIn+3])

                    widIn = int(content[i+2].find("width:"))+9
                    wid = int(content[i+1][widIn:widIn+3])

                    htIn = int(content[i+2].find("height:"))+9
                    ht = int(content[i+2][htIn:htIn+3])

                    rx = lx + wid
                    by = ty + ht

                    xA = max(xmin, lx)
                    yA = max(ymin, ty)
                    xB = min(xmax, rx)
                    yB = min(ymax, by)

                    # compute the area of intersection rectangle
                    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

                    # compute the area of both the prediction and ground-truth
                    # rectangles
                    boxAArea = abs((xmax - xmin) * (ymax - ymin))
                    boxBArea = abs((rx-lx) * (by - ty))

                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou = interArea / float(boxAArea + boxBArea - interArea)

                    lxIn = int(content[i+1].find("left_x:"))+9
                    lx = int(content[i+1][lxIn:lxIn+3])

                    tyIn = int(content[i+1].find("top_y:"))+8
                    ty = int(content[i+1][tyIn:tyIn+3])

                    widIn = int(content[i+1].find("width:"))+9
                    wid = int(content[i+1][widIn:widIn+3])

                    htIn = int(content[i+1].find("height:"))+9
                    ht = int(content[i+1][htIn:htIn+3])

                    rx = lx + wid
                    by = ty + ht

                    xA = max(xmin, lx)
                    yA = max(ymin, ty)
                    xB = min(xmax, rx)
                    yB = min(ymax, by)

                    # compute the area of intersection rectangle
                    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

                    # compute the area of both the prediction and ground-truth
                    # rectangles
                    boxAArea = abs((xmax - xmin) * (ymax - ymin))
                    boxBArea = abs((rx-lx) * (by - ty))

                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou2 = interArea / float(boxAArea + boxBArea - interArea)

                    if iou2 > iou:
                        iou = iou2
                    if iou >= 0.20:
                
                        confidence = float(content[i+1][7:9])/100
                        cases[case-1].append(confidence)
                        y_scores.append(confidence)
                    else:
                        y_scores.append(0)
                        


                elif content[i+1].find("polyp: ") >= 0:
                    lxIn = int(content[i+1].find("left_x:"))+9
                    lx = int(content[i+1][lxIn:lxIn+3])

                    tyIn = int(content[i+1].find("top_y:"))+8
                    ty = int(content[i+1][tyIn:tyIn+3])

                    widIn = int(content[i+1].find("width:"))+9
                    wid = int(content[i+1][widIn:widIn+3])

                    htIn = int(content[i+1].find("height:"))+9
                    ht = int(content[i+1][htIn:htIn+3])

                    rx = lx + wid
                    by = ty + ht

                    xA = max(xmin, lx)
                    yA = max(ymin, ty)
                    xB = min(xmax, rx)
                    yB = min(ymax, by)

                    # compute the area of intersection rectangle
                    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

                    # compute the area of both the prediction and ground-truth
                    # rectangles
                    boxAArea = abs((xmax - xmin) * (ymax - ymin))
                    boxBArea = abs((rx-lx) * (by - ty))

                    # compute the intersection over union by taking the intersection
                    # area and dividing it by the sum of prediction + ground-truth
                    # areas - the interesection area
                    iou = interArea / float(boxAArea + boxBArea - interArea)

                    if iou >= 0.20:
                
                        confidence = float(content[i+1][7:9])/100
                        cases[case-1].append(confidence)
                        y_scores.append(confidence)
                    else:
                        y_scores.append(0)

                if content[i+1].find("polyp: ") <= -1 and content[i+2].find("polyp: ") <= -1:
                    y_scores.append(0)

            elif case != 101:
                if content[i+1].find("polyp: ") >= 0:
                    confidence = float(content[i+1][7:9])/100
                    cases[case-1].append(confidence)
                    y_scores.append(confidence)

                else:
                    y_scores.append(0)
                    
            
            if case == 101:
                y_true.append(0)
                if content[i+1].find("polyp: ") >= 0:
                    confidence = float(content[i+1][7:9])/100
                    cases[case-1].append(confidence)
                    y_scores.append(confidence)
  
                else:
                    y_scores.append(0)
            else:
                y_true.append(1)
            
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.plot(fpr, tpr)
plt.ylabel("True Positive Rate (TPR)")
plt.xlabel("False Positive Rate (FPR)")
plt.savefig('ROCCurve.png')
plt.show()

y_pred = []


for i in range(len(y_scores)):
    if y_scores[i] > 0:
        y_pred.append(1)
    else:
        y_pred.append(0)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

precision = tp/(tp+fn)
recall = tp/(tp+fp)

f1 = 2*((precision*recall)/(precision+recall))

print(precision, recall, f1)


for i in range(len(cases)):
    if len(cases[i]) > 0:
        avcon.append(statistics.median(cases[i]))
    else:
        avcon.append(0)


#remove case 101 (non-polyp frames)
avcon.pop()


sizes = [3,18,6,4,3,3,6,12,4,3,5,5,5,3,5,4,4,2,3,3,3,2,12,15,7,5,5,2,13,4,12,15,5,3,15,7,7,5,13,5,3,7,10,5,3,2,5,3,3,10,15,6,4,4,3,4,5,6,8,8,6,7,7,3,3,6,5,15,3,15,4,5,3,5,3,3,4,12,4,10,6,3,13,5,8,4,3,4,5,10,13,7,7,6,8,5,15,4,5,3]

#5: IIA, 6: IS, 7: ISP, 8: IP

shape = [6,6,5,6,5,5,5,7,6,5,5,6,6,5,6,6,6,6,5,5,5,5,8,8,6,6,6,6,7,5,8,8,6,6,8,5,6,6,5,5,5,6,7,5,6,5,6,6,5,7,5,5,6,6,6,6,6,5,7,5,7,6,6,5,5,6,5,6,5,8,6,6,6,7,6,6,6,7,6,6,6,5,7,6,5,5,6,6,8,6,5,6,6,6,7,6,5,5,6,5]

#1. hyperplastic polyp, 2. sessile serated lesion, 3. low grade adenoma, 4. traditional serrated adenoma,  5. high grade adenoma, 6. invasive carcimona 

diagnosis = [3,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1,3,3,3,3,3,3,3,3,1,3,3,3,3,4,3,3,5,3,3,3,3,3,3,3,3,3,3,1,3,3,3,3,3,3,1,3,3,1,3,2,4,3,3,3,6,3,3,3,3,5,3,3,3,3,3,3,3,3,3,5,3,3,3,2,3,3,3,3,3,3,3,2,3,3,3,3,3,1,2,3,3,1]

#1: cecum, 2: rectum, 3: Ascending Colon, 4: Sigmoid Colon, 5: Transverse Colon, 6: Descending Colon

location = [1,2,3,4,5,4,6,4,4,5,6,2,5,4,5,5,4,4,5,3,4,3,3,4,4,6,3,5,4,4,6,3,4,3,4,4,5,3,3,5,2,5,3,4,3,5,5,5,5,4,1,4,2,4,3,4,5,5,4,5,3,3,2,4,1,4,5,2,6,4,5,5,1,4,5,1,3,4,6,4,4,4,2,6,3,4,1,3,6,3,3,6,6,4,4,4,1,5,4,2]


#split shape into confidence 
IIA = []
IS = []
ISP = []
IP = []

for i in range(0,len(avcon)):
    if shape[i] == 5:
        IIA.append(avcon[i])
    if shape[i] == 6:
        IS.append(avcon[i])
    if shape[i] == 7:
        ISP.append(avcon[i])
    if shape[i] == 8:
        IP.append(avcon[i])

#get average entropy for each shape

IIAAvEntropy = avEntropy(IIA)[1]
ISAvEntropy = avEntropy(IS)[1]
ISPAventropy = avEntropy(ISP)[1]
IPAventropy = avEntropy(IP)[1]

#plot graph for shape

entropys = [IIAAvEntropy,ISAvEntropy,ISPAventropy, IPAventropy]
# Numbers of pairs of bars you want
N = 4
# Specify the values of blue bars (height)

blue_bar = (statistics.median(IIA),statistics.median(IS),statistics.median(ISP),statistics.median(IP))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Shape')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of different Shaped Polyps')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('IIA', 'IS', 'ISP','IP'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()


#split into size ranges
size1 = []
size2 = []
size3 = []
size4 = []
size5 = []


for i in range(len(sizes)):
    if sizes[i] <= 3:
        size1.append(avcon[i])
    if sizes[i] > 3 and sizes[i] <= 6:
        size2.append(avcon[i])
    if sizes[i] > 6 and sizes[i] <= 9:
        size3.append(avcon[i])
    if sizes[i] > 9 and sizes[i] <= 12:
        size4.append(avcon[i])
    if sizes[i] > 12:
        size5.append(avcon[i])





#get entropy of size categories
S1AvEntropy = avEntropy(size1)[1]
S2AvEntropy = avEntropy(size2)[1]
S3AvEntropy = avEntropy(size3)[1]
S4AvEntropy = avEntropy(size4)[1]
S5AvEntropy = avEntropy(size5)[1]


#plot size graph

entropys = [S1AvEntropy,S2AvEntropy,S3AvEntropy,S4AvEntropy,S5AvEntropy]
# Numbers of pairs of bars you want
N = 5

# Specify the values of blue bars (height)

blue_bar = (statistics.median(size1),statistics.median(size2),statistics.median(size3),statistics.median(size4),statistics.median(size5))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Size (mm)')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of Different Sized Polyps')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('0-3','3-6','6-9','9-12','>12'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

#split into diagnosis categories
HP = []
SSL = []
LGA = []
TSA = []
HGA = []
IC = []



for i in range(len(diagnosis)):
    if diagnosis[i] == 1:
        HP.append(avcon[i])
    if diagnosis[i] == 2:
        SSL.append(avcon[i])
    if diagnosis[i] == 3:
        LGA.append(avcon[i])
    if diagnosis[i] == 4:
        TSA.append(avcon[i])
    if diagnosis[i] == 5:
        HGA.append(avcon[i])
    if diagnosis[i] == 6:
        IC.append(avcon[i])


#get diagnosis entropys
HPAvEntropy = avEntropy(HP)[1]
SSLAvEntropy = avEntropy(SSL)[1]
LGAAvEntropy = avEntropy(LGA)[1]
TSAAvEntropy = avEntropy(TSA)[1]
HGAAvEntropy = avEntropy(HGA)[1]
ICAvEntropy = avEntropy(IC)[1]

#plot diagnosis graph
entropys = [HPAvEntropy,SSLAvEntropy,LGAAvEntropy,TSAAvEntropy,HGAAvEntropy,ICAvEntropy]
# Numbers of pairs of bars you want
N = 6

# Specify the values of blue bars (height)

blue_bar = (statistics.median(HP),statistics.median(SSL),statistics.median(LGA),statistics.median(TSA),statistics.median(HGA),statistics.median(IC))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Diagnosis')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of Different Diagnosed Polyps')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Hyperplastic Polyp', 'Sessile Serrated Lesion','Low Grade Adenoma','Traditional Serrated Adenoma','High Grade Adenoma','Invasive Carcinoma'),rotation=15)

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

#sort into location categories

C = []
R = []
AC = []
SC = []
TC = []
DC = []

for i in range(len(shape)):
    if location[i] == 1:
        C.append(avcon[i])
    if location[i] == 2:
        R.append(avcon[i])
    if location[i] == 3:
        AC.append(avcon[i])
    if location[i] == 4:
        SC.append(avcon[i])
    if location[i] == 5:
        TC.append(avcon[i])
    if location[i] == 6:
        DC.append(avcon[i])


#get average entropy

CAvEntropy = avEntropy(C)[1]
RAvEntropy = avEntropy(R)[1]
ACAvEntropy = avEntropy(AC)[1]
SCAvEntropy = avEntropy(SC)[1]
TCAvEntropy = avEntropy(TC)[1]
DCAvEntropy = avEntropy(DC)[1]


#plot location graph

entropys = [CAvEntropy,RAvEntropy,ACAvEntropy,SCAvEntropy,TCAvEntropy,DCAvEntropy]
# Numbers of pairs of bars you want
N = 6

# Specify the values of blue bars (height)

blue_bar = (statistics.median(C),statistics.median(R),statistics.median(AC),statistics.median(SC),statistics.median(TC),statistics.median(DC))
# Specify the values of orange bars (height)
orange_bar = entropys

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(10,5))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, blue_bar , width, label='Average Confidence')
plt.bar(ind + width, orange_bar, width, label='Average Entropy')

plt.xlabel('Polyp Location')
plt.ylabel('Average Confidence/Entropy')
plt.title('Average Confidence and Entropy of Polyps From Different Locations')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('Cecum', 'Rectum','Ascending Colon','Sigmoid Colon','Transverse Colon','Descending Colon'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()

