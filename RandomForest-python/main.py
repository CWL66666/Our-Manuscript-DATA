import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



file_name_X_test = "DGA_testing_data.csv"
file_name_y_test = "DGA_testing_labels.csv"
file_name_X_train = "DGA_training_data.csv"
file_name_y_train = "DGA_training_labels.csv"

X_test = pd.read_csv(file_name_X_test,header = None)  #Read data
y_test = pd.read_csv(file_name_y_test,header = None)
X_train = pd.read_csv(file_name_X_train,header = None)
y_train = pd.read_csv(file_name_y_train,header = None)

X_train = np.array(X_train)
X_test = np.array(X_test)                 #Convert to array format
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)
# 4 Analyze the effect of the trained model on the test data
acc = model.score(X_test, y_test)  # Accuracy
T_sim = model.predict(X_test)
# 5 Run for 10 times and take the average value to prevent accidental events
Acc = 0  # Average accuracy index
for i in range(0, 10):  # Run for 10 times and take the average value
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    Acc = Acc + model.score(X_test, y_test)  # Accuracy accumulation
Acc = Acc/10  #Average accuracy

epochs = np.arange(1,77)
fig_1 = plt.figure()          #plot
ax1 = fig_1.add_subplot(1,2,1)
plt.plot(epochs,y_test,color='r',label='Test')
plt.plot(epochs,T_sim,'b*',label='T_sim')
plt.xlabel('Testing set sample number') #Represents the x-axis label
plt.ylabel('Failure Mode') #Represents the y-axis label
plt.title("Have ratio feature :"+str(format(acc*100,'.3f'))+'%' + '\n average accurate rate:' + str(format(Acc*100,'.3f'))+'%') #Icon Title Representation
plt.legend()
plt.show()

X_test = X_test[:,0:5]          #Original data
X_train = X_train[:,0:5]

model = RandomForestClassifier()
model.fit(X_train, y_train)
# 6 Analyze the effect of the trained model on the test data
acc = model.score(X_test, y_test)  # Accuracy
T_sim = model.predict(X_test)
# 7 Run for 10 times and take the average value to prevent accidental events
Acc = 0  # 
for i in range(0, 10):  # Run for 10 times and take the average value
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    Acc = Acc + model.score(X_test, y_test)  # accuracy
Acc = Acc/10  #Average accuracy
ax2 = fig_1.add_subplot(1,2,2)
plt.plot(epochs,y_test,color='r',label='Test')
plt.plot(epochs,T_sim,'b*',label='T_sim')
plt.xlabel('Testing set sample number')
plt.ylabel('Failure Mode')
plt.title("Original data:"+str(format(acc*100,'.3f'))+'%' + '\n average accurate rate:' + str(format(Acc*100,'.3f'))+'%') #Icon Title Representation
plt.legend()
plt.show()



