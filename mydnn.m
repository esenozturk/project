%% 
close all
clear all;clc;
load onlinedata  % Import Data

%Inputs and the output are defined here
inputs = @(T) [T.input1 T.input2 T.input3 T.input4 T.input5 T.input6 T.input7 T.input8 T.input9 T.input10 T.input11]; % 6 columns
output = @(T) [T.output];
C31s = struct;

%10% of the data separated for test
cv = cvpartition(height(Ta),'HoldOut',0.1);

T_train = Ta(cv.training,:);
T_test=Ta(cv.test,:);


%% Hyperparameters 
trainR = 0.8;
valR = 0.1;
testR = 0.1;
epoch = 2000;
max_fail = 6;

%% Define model
net = fitnet([32 8 4]); % 3 hidden layers were chosen for the analysis 
net.trainParam.epochs = epoch;
net.trainParam.max_fail = max_fail;
net.divideParam.trainRatio = trainR;
net.divideParam.valRatio = valR;
net.divideParam.testRatio = testR;
net.layers{4}.transferFcn = 'tansig'; % hyperbolic tangent sigmoid transfer function for all layers


% Make train array 
X_data = inputs(T_train);
y_data = output(T_train);

% Make test array 
X_test = inputs(T_test);
y_test = output(T_test);

%NaN values are eliminated
dummy1 = [X_data, y_data];
dummy1 = rmmissing(dummy1);
dummy2 = [X_test, y_test];
dummy2 = rmmissing(dummy2);

X_data = dummy1(:, 1:11);
y_data = dummy1(:, end);
X_test = dummy2(:, 1:11);
y_test = dummy2(:, end);

X_total = [X_data; X_test];
y_total = [y_data; y_test];

% Data scaling 
muX = mean(X_total);
sigX = std(X_total);
X_data = (X_data - muX) ./ sigX;
X_test = (X_test - muX) ./ sigX;

% Data scaling 
muy = mean(y_total);
sigy = std(y_total);
y_data = (y_data - muy) ./ sigy;
y_test = (y_test - muy) ./ sigy;

% net training 
X_data = X_data';
y_data = y_data';

[net, tr] = train(net, X_data, y_data, 'UseParallel', 'yes');

%for training
y_pred_train = net(X_data);

% Regression for training
figure(1)
plotregression(y_data, y_pred_train);
title('Regression Plot for Training Set','FontSize', 14);

% Fit for training
figure(2)
plot(y_data,'+');
hold on;
plot(y_pred_train);
hold off;
legend({'Real','Predicted'})
title('Training Set for ANN','FontSize', 14);
xlabel('Observation','FontSize', 12);
ylabel('Concentration' ,'FontSize', 12);
ylim([-1.8 2])

% R^2 value
stats=rs_stats(y_pred_train', y_data')


%for test
X_test = X_test';
y_test = y_test';
y_pred_test = net(X_test);

% Regression for test
figure(3)
plotregression(y_test, y_pred_test);
title('Regression Plot for Test Set','FontSize', 14);

% Fit for test
figure(4)
plot(y_test,'+');
hold on;
plot(y_pred_test);
hold off;
legend({'Real','Predicted'})
title('Test Set for ANN','FontSize', 14);
xlabel('Observation','FontSize', 12);
ylabel('Concentration' ,'FontSize', 12);
ylim([-1.8 2])

%R^ value for test
stats_test=rs_stats(y_pred_test', y_test')

save mydnn net tr


% CROSS VALIDATION RESULTS

figure(5)
R2_train =[0.9875 0.9871 0.9881 0.9875 0.9871 0.9881 0.9878 0.9875 0.9875 0.9875 0.9871 0.9881 0.9878 0.9870 0.9876 0.9869 0.9868 0.9875 0.9880  ];
R2_test  =[0.9866 0.9866 0.9869 0.9866 0.9869 0.9869 0.9870 0.9877 0.9866 0.9866 0.9866 0.9869 0.9870 0.9852 0.9866 0.9868 0.9875 0.9880 0.9877 0.9880  ];

graph1= plot(R2_train, 'r')
set(graph1,'LineWidth',2);
xlabel('Number of Runs','FontSize', 14) 
ylabel('R^{2} value','FontSize', 14) 
hold on
graph2=plot(R2_test, 'b')
set(graph2,'LineWidth',2);
legend ('R^{2} train', 'R^{2} test')
title('Cross Validation Results','FontSize', 14);


