# MA305-Bitcoin-Final-Project
Bitcoin Historical Price Analysis and Prediciton

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:24:14 2021

@author: 15416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.close("all")

df = pd.read_excel (r'C:\Users\15416\Desktop\Bitcoin Analysis\Bitcoin_HistoricalData.xlsx')
print(df)

allopendata= df[["Date", "Open"]]

print(df['Date'])

df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')



df.set_index('Date', inplace = True)

print(df)

while True:
    a = input('What would you like to put for the first input? \n(You must type using format YYYY-MM-DD): ')
    if a < '2014' or a >= '2022':
        print('\nThat input is invalid, please try retyping your input.')
    else:
        break
    
while True:
    b = input('What would you like to put for the second input? \n(You must type using format YYYY-MM-DD): ')
    if b < '2014' or b < a or b >= '2022':
        print('\nThat input is invalid, please try retyping your input.')
    else:
        break


df.loc[a:b,"Open"].plot(linestyle='-')
df.loc[a:b,"Close"].plot(linestyle='--')

plt.legend(loc='upper left')
plt.show()

mean1 = df[a:b]['Open'].mean()
mean2 = df[a:b]['Close'].mean()
print('\nThe mean of the opening values in this selected range is %s.'%(mean1))
print('The mean of the closing values in this selected range is %s.' %(mean2))

highest_high = df[a:b]['High'].max()
lowest_low = df[a:b]['Low'].min()
print("\nThe highest price identified within that time range was %s, while the lowest price identified was %s." % (highest_high,lowest_low))




df = pd.DataFrame(allopendata,columns=['Date','Open'])
print(df)
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

#Year Breakdown into plots of open price over time
Year17= allopendata[837:1201]
print(Year17)
df = pd.DataFrame(Year17,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

Year18= allopendata[1202:1567]
print(Year18)
df = pd.DataFrame(Year18,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

Year19= allopendata[1567:1932]
print(Year19)
df = pd.DataFrame(Year19,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

Year20= allopendata[1932:2298]
print(Year20)
df = pd.DataFrame(Year20,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

#January Monthly breakdown
Jan17= allopendata[837:868]
print(Jan17)
df = pd.DataFrame(Jan17,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

Jan18= allopendata[1202:1233]
print(Jan18)
df = pd.DataFrame(Jan18,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()


Jan19= allopendata[1567:1598]
print(Jan19)
df = pd.DataFrame(Jan19,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()

Jan20= allopendata[1932:1963]
print(Jan20)
df = pd.DataFrame(Jan20,columns=['Date','Open'])
df.plot(x ='Date', y='Open', kind = 'line')
plt.show()


#Extracting days 1 year appart and comparing for statistical significance

Day17 = Year17.iloc[0]['Open']
Day18 = Year18.iloc[0]['Open']
Day19 = Year19.iloc[0]['Open']
Day20 = Year20.iloc[0]['Open']

diff1718 = Day17 - Day18
diff1719 = Day17 - Day19
diff1720 = Day17 - Day20

print(diff1718)
print(diff1719)
print(diff1720)

#User Interface for Prediction Model
#daterange = input('\nEnter a Date to predict the opening price 10 days later: ')
#Implementing the grey model method
Jan17Open=Jan17.iloc[:,1]
print(Jan17Open)
print("\n")
#January 2017 Predictive model Step 1 X and Z value generations FIRST TEN DAYS
#Accumulating Generation Operator Sequence X
Xvalue_list17 = []
n=0
Xvalue=0
while n <= 10:
    value = Jan17.iloc[n,1]
    Xvalue = value + Xvalue
    Xvalue_list17.append(Xvalue)
    n=n+1
    
print(Xvalue_list17)
print("\n")
#z is the Mean Gereration of Consecutive Neighbors of Sequence X
Zvalue_list17 = []
a = 1
while a <= 10:
    Zvalue = 0.5*(Xvalue_list17[a]+Xvalue_list17[a-1])
    Zvalue_list17.append(Zvalue)
    a = a + 1

print(Zvalue_list17)
print("\n")
#Change X list to Y array 1 column
Y17list = []
n=0
while n<=9:
    Y17value = Jan17.iloc[n,1]
    Y17list.append(Y17value)
    n=n+1
print(Y17list)
Y17 = np.asarray(Y17list)
print('Y17',Y17)
print("\n")

#Change Z list to B array, make values negative, add column of 1s
B17 = np.asarray(Zvalue_list17)
#B17 = B17.astype(int)
B17 = np.negative(B17)
ones = np.array([1,1,1,1,1,1,1,1,1,1])
B17 = np.vstack([B17,ones])
B17 = np.transpose(B17)
print('B17', B17)
print("\n")


#Compute the parameters a and b
B17T = np.transpose(B17)
BtB = np.dot(B17T, B17)
#BtB = BtB.astype(int)
BtB_inv = np.linalg.inv(BtB) 
#BtB_inv = BtB_inv.astype(int)
BtY = np.dot(B17T, Y17)
#BtY = np.absolute(BtY)
AB = np.dot(BtB_inv, BtY)
#AB = AB.astype(int)
print('AB', AB)
print("\n")

#b min and max value calculaitons
a = AB[0]
b = AB[1]
start = 1
while start <= 9:
    x = Y17list[start]
    z = Zvalue_list17[start]
    az = a*z
    bn = az +x
    bn = bn/(start+1)
    bn = bn.astype(int)
#    print('B',start, bn)
    start = start+1

#one ahead prediction
xhat_one = (963.63-(b/a))*math.exp(-a*1)+(b/a) 
print(xhat_one)
xhat_one = (998.63-(b/a))*math.exp(-a*2)+(b/a) 
print(xhat_one)
xhat_one = (1021.60-(b/a))*math.exp(-a*3)+(b/a) 
print(xhat_one)


n = 1
m = 0
print('January 1st - 9th, 2017')
print("{:<8} {:<15} {:<10} {:<10}".format('\n\nN', 'Prediciton', 'Actual', 'Percent Error'))
while n <= 9:
    previous = Y17list[m]
    prediction = (1 - math.exp(a))*(previous-(b/a))*math.exp(-a*n)
    prediction = prediction.astype(int)
    error = abs((prediction-Y17list[n].astype(int))/Y17list[n].astype(int))*100
    error = round(error, 2)
    print("{:<8} {:<15} {:<10} {:<10}".format(n, prediction, Y17list[n].astype(int),error))
    n=n+1
    m=m+1









"""January 2018"""

Jan18Open=Jan18.iloc[:,1]
#print(Jan18Open)
print("\n")
#January 2018 Predictive model Step 1 X and Z value generations FIRST TEN DAYS
#Accumulating Generation Operator Sequence X
Xvalue_list18 = []
n=0
Xvalue=0
while n <= 10:
    value = Jan18.iloc[n,1]
    Xvalue = value + Xvalue
    Xvalue_list18.append(Xvalue)
    n=n+1
    
#print(Xvalue_list18)
print("\n")
#z is the Mean Gereration of Consecutive Neighbors of Sequence X
Zvalue_list18 = []
a = 1
while a <= 10:
    Zvalue = 0.5*(Xvalue_list18[a]+Xvalue_list18[a-1])
    Zvalue_list18.append(Zvalue)
    a = a + 1

#print(Zvalue_list18)
print("\n")
#Change X list to Y array 1 column
Y18list = []
n=0
while n<=9:
    Y18value = Jan18.iloc[n,1]
    Y18list.append(Y18value)
    n=n+1
#print(Y18list)
Y18 = np.asarray(Y18list)
#print('Y18',Y18)
#print("\n")

#Change Z list to B array, make values negative, add column of 1s
B18 = np.asarray(Zvalue_list18)
B18 = np.negative(B18)
ones = np.array([1,1,1,1,1,1,1,1,1,1])
B18 = np.vstack([B18,ones])
B18 = np.transpose(B18)
#print('B18', B18)
#print("\n")


#Compute the parameters a and b
B18T = np.transpose(B18)
BtB = np.dot(B18T, B18)
#BtB = BtB.astype(int)
BtB_inv = np.linalg.inv(BtB) 
#BtB_inv = BtB_inv.astype(int)
BtY = np.dot(B18T, Y18)
#BtY = np.absolute(BtY)
AB = np.dot(BtB_inv, BtY)
#AB = AB.astype(int)
print('AB', AB)
print("\n")

#b min and max value calculaitons
a = AB[0]
b = AB[1]
start = 1
while start <= 9:
    x = Y18list[start]
    z = Zvalue_list18[start]
    az = a*z
    bn = az +x
    bn = bn/(start+1)
    bn = bn.astype(int)
#    print('B',start, bn)
    start = start+1

#one ahead prediction
xhat_one = (Y18list[0]-(b/a))*math.exp(-a*1)+(b/a) 
#print(xhat_one)
xhat_one = (Y18list[1]-(b/a))*math.exp(-a*2)+(b/a) 
#print(xhat_one)
xhat_one = (Y18list[2]-(b/a))*math.exp(-a*3)+(b/a) 
#print(xhat_one)


n = 1
m = 0
print('January 1st - 9th, 2018')
print("{:<8} {:<15} {:<10} {:<10}".format('\n\nN', 'Prediciton', 'Actual', 'Percent Error'))
while n <= 9:
    previous = Y18list[m]
    prediction = (1 - math.exp(a))*(previous-(b/a))*math.exp(-a*n)
    prediction = prediction.astype(int)
    error = abs((prediction-Y18list[n].astype(int))/Y18list[n].astype(int))*100
    error = round(error, 2)
    print("{:<8} {:<14} {:<10} {:<10}".format(n, prediction, Y18list[n].astype(int),error))
    n=n+1
    m=m+1
    
    
    



"""January 2019"""
Jan19Open=Jan19.iloc[:,1]
#print(Jan18Open)
print("\n")
#January 2019 Predictive model Step 1 X and Z value generations FIRST TEN DAYS
#Accumulating Generation Operator Sequence X
Xvalue_list19 = []
n=0
Xvalue=0
while n <= 10:
    value = Jan19.iloc[n,1]
    Xvalue = value + Xvalue
    Xvalue_list19.append(Xvalue)
    n=n+1
    
#print(Xvalue_list19)
print("\n")
#z is the Mean Gereration of Consecutive Neighbors of Sequence X
Zvalue_list19 = []
a = 1
while a <= 10:
    Zvalue = 0.5*(Xvalue_list19[a]+Xvalue_list19[a-1])
    Zvalue_list19.append(Zvalue)
    a = a + 1

#print(Zvalue_list19)
print("\n")
#Change X list to Y array 1 column
Y19list = []
n=0
while n<=9:
    Y19value = Jan19.iloc[n,1]
    Y19list.append(Y19value)
    n=n+1
#print(Y19list)
Y19 = np.asarray(Y19list)
#print('Y19',Y19)
#print("\n")

#Change Z list to B array, make values negative, add column of 1s
B19 = np.asarray(Zvalue_list19)
B19 = np.negative(B19)
ones = np.array([1,1,1,1,1,1,1,1,1,1])
B19 = np.vstack([B19,ones])
B19 = np.transpose(B19)
#print('B19', B19)
#print("\n")


#Compute the parameters a and b
B19T = np.transpose(B19)
BtB = np.dot(B19T, B19)
#BtB = BtB.astype(int)
BtB_inv = np.linalg.inv(BtB) 
#BtB_inv = BtB_inv.astype(int)
BtY = np.dot(B19T, Y19)
#BtY = np.absolute(BtY)
AB = np.dot(BtB_inv, BtY)
#AB = AB.astype(int)
print('AB', AB)
print("\n")

#b min and max value calculaitons
a = AB[0]
b = AB[1]
start = 1
while start <= 9:
    x = Y19list[start]
    z = Zvalue_list19[start]
    az = a*z
    bn = az +x
    bn = bn/(start+1)
    bn = bn.astype(int)
#    print('B',start, bn)
    start = start+1

#one ahead prediction
xhat_one = (Y19list[0]-(b/a))*math.exp(-a*1)+(b/a) 
#print(xhat_one)
xhat_one = (Y19list[1]-(b/a))*math.exp(-a*2)+(b/a) 
#print(xhat_one)
xhat_one = (Y19list[2]-(b/a))*math.exp(-a*3)+(b/a) 
#print(xhat_one)


n = 1
m = 0
print('January 1st - 9th, 2019')
print("{:<8} {:<15} {:<10} {:<10}".format('\n\nN', 'Prediciton', 'Actual', 'Percent Error'))
while n <= 9:
    previous = Y19list[m]
    prediction = (1 - math.exp(a))*(previous-(b/a))*math.exp(-a*n)
    prediction = prediction.astype(int)
    error = abs((prediction-Y19list[n].astype(int))/Y19list[n].astype(int))*100
    error = round(error, 2)
    print("{:<8} {:<14} {:<10} {:<10}".format(n, prediction, Y19list[n].astype(int),error))
    n=n+1
    m=m+1
    
    
    
    
    
"""January 2020"""
Jan20Open=Jan20.iloc[:,1]
#print(Jan20Open)
print("\n")
#January 2020 Predictive model Step 1 X and Z value generations FIRST TEN DAYS
#Accumulating Generation Operator Sequence X
Xvalue_list20 = []
n=0
Xvalue=0
while n <= 10:
    value = Jan20.iloc[n,1]
    Xvalue = value + Xvalue
    Xvalue_list20.append(Xvalue)
    n=n+1
    
#print(Xvalue_list20)
print("\n")
#z is the Mean Gereration of Consecutive Neighbors of Sequence X
Zvalue_list20 = []
a = 1
while a <= 10:
    Zvalue = 0.5*(Xvalue_list20[a]+Xvalue_list20[a-1])
    Zvalue_list20.append(Zvalue)
    a = a + 1

#print(Zvalue_list20)
print("\n")
#Change X list to Y array 1 column
Y20list = []
n=0
while n<=9:
    Y20value = Jan20.iloc[n,1]
    Y20list.append(Y20value)
    n=n+1
#print(Y20list)
Y20 = np.asarray(Y20list)
#print('Y19',Y19)
#print("\n")

#Change Z list to B array, make values negative, add column of 1s
B20 = np.asarray(Zvalue_list20)
B20 = np.negative(B20)
ones = np.array([1,1,1,1,1,1,1,1,1,1])
B20 = np.vstack([B20,ones])
B20 = np.transpose(B20)
#print('B20', B20)
#print("\n")


#Compute the parameters a and b
B20T = np.transpose(B20)
BtB = np.dot(B20T, B20)
#BtB = BtB.astype(int)
BtB_inv = np.linalg.inv(BtB) 
#BtB_inv = BtB_inv.astype(int)
BtY = np.dot(B20T, Y20)
#BtY = np.absolute(BtY)
AB = np.dot(BtB_inv, BtY)
#AB = AB.astype(int)
print('AB', AB)
print("\n")

#b min and max value calculaitons
a = AB[0]
b = AB[1]
start = 1
while start <= 9:
    x = Y20list[start]
    z = Zvalue_list20[start]
    az = a*z
    bn = az +x
    bn = bn/(start+1)
    bn = bn.astype(int)
#    print('B',start, bn)
    start = start+1

#one ahead prediction
xhat_one = (Y20list[0]-(b/a))*math.exp(-a*1)+(b/a) 
#print(xhat_one)
xhat_one = (Y20list[1]-(b/a))*math.exp(-a*2)+(b/a) 
#print(xhat_one)
xhat_one = (Y20list[2]-(b/a))*math.exp(-a*3)+(b/a) 
#print(xhat_one)


n = 1
m = 0
print('January 1st - 9th, 2020')
print("{:<8} {:<15} {:<10} {:<10}".format('\n\nN', 'Prediciton', 'Actual', 'Percent Error'))
while n <= 9:
    previous = Y20list[m]
    prediction = (1 - math.exp(a))*(previous-(b/a))*math.exp(-a*n)
    prediction = prediction.astype(int)
    error = abs((prediction-Y20list[n].astype(int))/Y20list[n].astype(int))*100
    error = round(error, 2)
    print("{:<8} {:<14} {:<10} {:<10}".format(n, prediction, Y20list[n].astype(int),error))
    n=n+1
    m=m+1
