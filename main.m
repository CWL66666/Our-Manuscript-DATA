%If ¡®Function or variable 'classrf' is not defined_ train¡¯ error occurs during operation, 
%We can send RF_ Class_ Add C file to path
%% I. Clear environment variables
clear all
clc

%% III. Import data
training_data=csvread('DGA_training_data.csv');
training_labels=csvread('DGA_training_labels.csv');
testing_data=csvread('DGA_testing_data.csv');
testing_labels=csvread('DGA_testing_labels.csv');

% Training set
P_train = training_data;
T_train = training_labels;
% Testing set
P_test = testing_data;
T_test = testing_labels;

T_test_num=size(T_test,1);%%Number of test set data
T_train_num=size(T_train,1);%%Number of train set data

%% III. Running results of ratio feature
model = classRF_train(P_train,T_train,200);
[T_sim,votes] = classRF_predict(P_test,model);
accuracy = length(find(T_sim ==T_test ))/T_test_num;  %Accuracy of one-time operation

accuracy_time = zeros(1,20);                  %Run 20 times and take the average to prevent accidental occurrence
for k = 1:length(accuracy_time)
    % Create random forest
    model = classRF_train(P_train,T_train,200);
    % Simulation test
    T_sim = classRF_predict(P_test,model);
    accuracy_time(k) = length(find(T_sim == T_test)) / length(T_test);
end
Accuracy = mean(accuracy_time)       

subplot(1,2,1);
T_test_num=size(testing_labels,1);
plot(1:T_test_num,testing_labels','r')
hold on
plot(1:T_test_num,T_sim','bo')
hold off
string = {'RandomRorest Have ratio feature';
          [ num2str(accuracy*100) '%']
               'average accuracy';
          [num2str(Accuracy*100) '%'] };
title(string)



%% IV. Running results of Original data
P_train = P_train(:,1:5);
P_test = P_test(:,1:5);         %Original data

model = classRF_train(P_train,T_train,200);
[T_sim,votes] = classRF_predict(P_test,model);
accuracy = length(find(T_sim ==T_test ))/T_test_num;   %Accuracy of one-time operation

accuracy_time = zeros(1,20);                  %Run 20 times and take the average to prevent accidental occurrence
for k = 1:length(accuracy_time)
    % Create random forest
    model = classRF_train(P_train,T_train,200);
    % Simulation test
    T_sim = classRF_predict(P_test,model);
    accuracy_time(k) = length(find(T_sim == T_test)) / length(T_test);
end
Accuracy = mean(accuracy_time)       

subplot(1,2,2);
T_test_num=size(testing_labels,1);
plot(1:T_test_num,testing_labels','r')
hold on
plot(1:T_test_num,T_sim','bo')
hold off

string = {'RandomRorest Original data';
          [ num2str(accuracy*100) '%']
               'average accuracy';
          [num2str(Accuracy*100) '%'] };
title(string)
   
