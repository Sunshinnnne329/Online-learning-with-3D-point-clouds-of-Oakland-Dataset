%% Online gradient descent on 3D point clouds of Oakland
% CS 8803 Statistical Techniques in Robotics, Lab 2: Online Learning
% Chaitanya Maniar, Sampada Upasani
%% load the data from files
clc; clear all; close all; 

tempdata_train = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
tempdata_test = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

% Class label definitions: veg, wire, pole, ground, facade

classes = [1004, 1100, 1103, 1200, 1400];

c1=find(tempdata_train(:,5)==classes(1));
c2=find(tempdata_train(:,5)==classes(2));
c3=find(tempdata_train(:,5)==classes(3));
c4=find(tempdata_train(:,5)==classes(4));
c5=find(tempdata_train(:,5)==classes(5));

%display([length(c1),length(c2),length(c3),length(c4),length(c5)]);

dataadd1=tempdata_train(c1,:);    
dataadd2=tempdata_train(c2,:);
dataadd3=tempdata_train(c3,:);
dataadd4=tempdata_train(c4,:);
dataadd5=tempdata_train(c5,:);

data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);
errdata = zeros(1,5);



% % Add more data for number of data points to be the same for SD to perform better
for p=1:8
    data1=[data1;dataadd1];
end

for p=1:80
    data1=[data1;dataadd2];
end
for p=1:47
    data1=[data1;dataadd3];
end

for p=1:5
    data1=[data1;dataadd5];
end

c1=find(data1(:,5)==classes(1));
c2=find(data1(:,5)==classes(2));
c3=find(data1(:,5)==classes(3));
c4=find(data1(:,5)==classes(4));
c5=find(data1(:,5)==classes(5));

%display([length(c1),length(c2),length(c3),length(c4),length(c5)]);

%Randomize the data
data1=data1(randsample(1:length(data1),length(data1)),:);

data2=data2(randsample(1:length(data2),length(data2)),:);

% First dataset
x1 = data1(:,1);        % x position
y1 = data1(:,2);        % y position
z1 = data1(:,3);        % z position
c1 = data1(:,4);        % indices
l1 = data1(:,5);        % labels
f1 = data1(:,6:end);    % features
%f1_err = [f1 rand(length(f1),5)];

% Second dataset
x2 = data2(:,1);        % x position
y2 = data2(:,2);        % y position
z2 = data2(:,3);        % z position
c2 = data2(:,4);        % indices
l2 = data2(:,5);        % labels
f2 = data2(:,6:end);    % features
%f2_err = [f2 rand(length(f2),5)];

%% Initialize online gradient descent

% Define our training data
training_data = data1;
l_training = l1;
f_training = f1;
T1 = length(training_data);

% Define our test data
test_data = data2;
l_test = l2;
f_test = f2;
T2 = length(test_data);


% Define number of features / classes
F = 10;     % number of features
C = 5;      % number of classes
labels1 = zeros(T1,C);
labels2 = zeros(T2,C);

% Define +1 or -1 labels for data

% for c = 1:C
%     if(l_training == classes(c))
%         labels1(:,c) = 1;
%     else 
%         labels1(:,c) = -1;
%     end
%     if(l_test == classes(c))
%         labels2(:,c) = 1;
%     else 
%         labels2(:,c) = -1;
%     end
% end
for c = 1:C
    labels1(:,c) = ((l_training == classes(c)) - 0.5)*2;
    labels2(:,c) = ((l_test == classes(c)) - 0.5)*2;
end

% Initialize weights for each class (each column is a different class)
w = zeros(size(f_training,2),C); % initialize weights


%% Run online gradient descent

%Iterate this many times through the training data
tic
for iter = 1:1
    % Iterate through training data
    for t = 1:T1

        for i = 1:C
            grad = 2*(w(:,i)'*f_training(t,:)'-labels1(t,i))*f_training(t,:)'; %Gradient of w'x^2-y
            alpha = 1/(sqrt(T1)); % Step size chosen to minimize regret
            w(:,i) = w(:,i) - alpha*grad;

        end
    end
    display(iter);
end
toc

%% Predict the test data
% Define the correct labels for the test data
tic
pred_label = zeros(T2,1);
error = zeros(C,1);
% Predict on test data
for t = 1:T2
    % Find which class this point belongs to and record it
    class = find(labels2(t,:) == 1);
    nc = find(labels2(t,:) == -1);

    [~, pred_label(t)] = max(w(:,1:C)'*f_test(t,:)');
    
    if pred_label(t) ~= class
        error(class) = error(class) + 1;
    end
end
total_classes = sum(labels2 == 1);

% Display the error of the test data
total_error = sum(error)/T2;
error = error ./ total_classes';
disp('Total error classwise:')
disp(error*100)
fprintf('Total Error = %f %',total_error*100)
toc
%% Write data to a file of the same format as the original
file = fopen('data1OGD.pcd','w');

if (file < 0)
    error('Could not open file');
end

fprintf(file,strcat('#\n#\n#\n'));

for i = 1:T2
    % Set the color for each class
    fprintf(file, '%f %f %f %d %d \n', x2(i), y2(i), z2(i), i, classes(pred_label(i)));
end

fclose(file);














