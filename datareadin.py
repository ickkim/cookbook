# to do: zip(x,y)
# normalizing is not so impressive. make it in the range of (-1,1)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

xloc = '/home/sr6/sungick777.kim/PEITS_root/PEITS/data/analysis/xdata/'
yloc = '/home/sr6/sungick777.kim/PEITS_root/PEITS/data/analysis/ydata/downsized/directmade/'

i = 0
# xdata = pd.Series();
xdata = pd.DataFrame();
ydata = []
for f in sorted(os.listdir(xloc)):
    if f[-12:-9] == 'rot' and f[-7:] == 'pad.txt':
        # print([xloc+f[:-15]])
        # print([xloc+f])
        for g in sorted(os.listdir(yloc)):

            if g[-22:-19] == 'rot' and g[-17:] == '_conductivity.txt':
                # print([yloc+g[:-23]])
                if f[:-15] == g[:-23] and f[-12:-7] == g[-22:-17]:
                    # print([xloc+f[:-15]])
                    # print([xloc+f])
                    # print([yloc+g[:-23]])
                    # print([yloc+g])
                    xf = []
                    xf = pd.read_csv(xloc + f, names=['voltage'])
                    # xdata = xdata.append(xf['voltage'], ignore_index=True)
                    xdata = pd.concat([xdata, xf], axis=1, ignore_index=True,
                                      verify_integrity=['voltage'])  # ignore_index=False)
                    # print(xdataframe.head())
                    yf = []
                    # yf = pd.read_csv(yloc+g, names=['xloc', 'yloc','sigma'])
                    yf = np.genfromtxt(yloc + g, delimiter=',')
                    # [A,B]=yf.shape
                    # if A == 1306:
                    #    print(i)
                    # yf = yf.stack()
                    ydata.append(yf[:, 2])
                    # print(ydataframe.head())
                    i = i + 1  # check the n
print('total matched set is ' + repr(i))

### input data normalizing
xdata = xdata.T
xdata = np.array(xdata)
##scaler = MinMaxScaler().fit(xdata)
##scaler = MaxAbsScaler().fit(xdata)

#scaler = Normalizer().fit(xdata)
scaler = StandardScaler().fit(xdata)
normalizedX = scaler.transform(xdata)


plt.clf()
plt.plot(range(0, 256), normalizedX[1][:], label='xdata1')
plt.plot(range(0, 256), normalizedX[201][:], label='xdata8')
plt.show()

### output data -> 0 or 1, binary classification problem
ydata = np.array(ydata)
(ylen, ywid) = ydata.shape

for i in range(ylen):
    for j in range(ywid):
        if ydata[i, j] > 0.4975:
            ydata[i, j] = 0
        else:
            ydata[i, j] = 1

'''
i = 99
plt.clf()
plt.scatter(yf[:, 0], yf[:, 1], c=ydata[i, :])
plt.show()
'''
