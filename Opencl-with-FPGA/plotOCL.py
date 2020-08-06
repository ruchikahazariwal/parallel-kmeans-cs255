import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


xFile = "Data/X1000000.txt"
yFile = "Data/Y1000000.txt"
cFile = "OCLC.txt"
cxFile = "OCLCX.txt"
cyFile = "OCLCY.txt"

fX = open(xFile, "r")
fY = open(yFile, "r")
fC = open(cFile, "r")
fCX = open(cxFile, "r")
fCY = open(cyFile, "r")
contentsX = fX.read()
contentsY = fY.read()
contentsC = fC.read()
contentsCX = fCX.read()
contentsCY = fCY.read()

contentsX = contentsX.split("\n")
contentsY = contentsY.split("\n")
contentsC = contentsC.split("\n")
contentsCX = contentsCX.split("\n")
contentsCY = contentsCY.split("\n")

X = np.asfarray(contentsX, float)
Y = np.asfarray(contentsY, float)
C = np.asarray(contentsC, int)
cX = np.asfarray(contentsCX, float)
cY = np.asfarray(contentsCY, float)

colours = ["#f44242", "#f4c741", "#6df441",
           "#41d3f4", "#4143f4", "#9a41f4", "#f441b5", "#f49542"]

for i in C:
    idx = C == i
    #print(idx)
    a = np.where(idx)
    x = X[a]
    y = Y[a]
    plt.scatter(x, y, c="red", s=7)
    plt.scatter(cX, cY, c="black", s=7)

#a = C[np.where(C == 0)]
#a = C.index(1)
# print(a)
#plt.scatter(X, Y, c='black', s=7);
plt.xlabel("X-axis") 
plt.ylabel("Y-axis") 
plt.show()
plt.savefig('FPGAChart.png')
