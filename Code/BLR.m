%% BLR on 3D point clouds of Oakland
% CS 8803 Statistical Techniques in Robotics, Lab 2: Online Learning
% Chaitanya Maniar, Sampada Upasani

%% Load the data
clc; clear all; close all;

% Data 1 is training data
data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
% Data2 is test data
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

% Randomizing the data
data1=data1(randsample(1:length(data1),length(data1)),:);
data2=data2(randsample(1:length(data2),length(data2)),:);

% Class label definitions: veg, wire, pole, ground, facade
classes = [1004, 1100, 1103, 1200, 1400];

% Training data columns
x1 = data1(:,1);        % x position
y1 = data1(:,2);        % y position
z1 = data1(:,3);        % z position
i1 = data1(:,4);        % indices
l1 = data1(:,5);        % labels
f1 = data1(:,6:end);    % features
% f1_err = [f1 rand(length(f1),5)];

% Test data columns
x2 = data2(:,1);
y2 = data2(:,2);
z2 = data2(:,3);
i2 = data2(:,4);
l2 = data2(:,5);
f2 = data2(:,6:end);
% f2_err = [f2 rand(length(f2),5)];

% Number of features / classes
len_1 = length(l1);
len_2 = length(l2);
len_F = length(f2(1,:));

%% Set up training data and test data
l_train = [];
f_train = [];
x_test = [];
y_test = [];
z_test = [];
l_test = [];
f_test = [];

% Class label definitions: veg, wire, pole, ground, facade
% classes = [1004, 1100, 1103, 1200, 1400];

class1 = 1004;
class2 = 1100;

%Generating labels for training data
for i = 1:len_1
    if (l1(i) == class1)
        l_train(end+1) = 1;
        f_train(end+1,:) = f1(i,:);
    elseif (l1(i) == class2)
        l_train(end+1) = -1;
        f_train(end+1,:) = f1(i,:);
    end
end

%Generating labels for test data

for i = 1:len_2
    if (l2(i) == class1)
        l_test(end+1) = 1;
        x_test(end+1,:) = x2(i,:);
        y_test(end+1,:) = y2(i,:);
        z_test(end+1,:) = z2(i,:);
        f_test(end+1,:) = f2(i,:);
    elseif (l2(i) == class2)
        x_test(end+1,:) = x2(i,:);
        y_test(end+1,:) = y2(i,:);
        z_test(end+1,:) = z2(i,:);
        l_test(end+1) = -1;
        f_test(end+1,:) = f2(i,:);
    end
end

% Define length training data
TR = length(f_train);

% Define length test data
TE = length(f_test);

%% Standardize the features for training data
f_train = f_train - repmat(mean(f_train),TR,1);
f_train = f_train ./ repmat(std(f_train),TR,1);

for i = 1:TR
    if (isnan(f_train(i,end)))
        f_train(i,end) = 1;
    end
end

%% Standardize the features for training data
f_test = f_test - repmat(mean(f_test),TE,1);
f_test = f_test ./ repmat(std(f_test),TE,1);

for i = 1:TE
    if (isnan(f_test(i,end)))
        f_test(i,end) = 1;
    end
end

%% Applying Bayesian Linear Regression

% Initialize Js
J = zeros(len_F,1);

P = inv(eye(len_F));
sigma = 1;
tic
% Update equations in natural parametrization(Training of data)
for i = 1:TR
    P = P + 1/sigma^2*f_train(i,:)*f_train(i,:)';
    J = J + 1/sigma^2*l_train(i)*f_train(i,:)';
end

w = inv(P)*J;
toc
%% Test the data
tic
pred_label = zeros(TE,1);
err = zeros(2,1);
labeled_test = zeros(TE,1);
% Predict on test data
for t = 1:TE
    pred_label(t) = sign(dot(w,f_test(t,:)));
    
    if pred_label(t) ~= l_test(t)
        if l_test(t) == 1
            err(1) = err(1) + 1;
        else
            err(2) = err(2) + 1;
        end
    end
end
total_classes(1) = sum(l_test == 1);
total_classes(2) = sum(l_test == -1);

% Display the error of the test data
total_error = sum(err)/TE;
err = err ./ total_classes';
err
toc
%% Visualizing the data
file = fopen('data1BLR.pcd','w');

if (file < 0)
    err('Could not open file');
end

fprintf(file,strcat('#\n#\n#\n'));

for i = 1:TE
    % Get the right class
    if pred_label(i) == 1
        class = class1;
    else
        class = class2;
    end
    fprintf(file, '%f %f %f %d %d \n', x_test(i), y_test(i), z_test(i), i, class);
end

fclose(file);
























