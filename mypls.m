%%
clear;clc;
close all
load onlinedata  % Import data

%Inputs and outputs are defined here.
inputs = @(T) [T.input1 T.input2 T.input3 T.input4 T.input5 T.input6 T.input7 T.input8 T.input9 T.input10 T.input11]; % 6 columns
output = @(T) [T.output];
C31s = struct;

% 10% of the data is separated for test.
cv = cvpartition(height(Ta),'HoldOut',0.1);

T_train = Ta(cv.training,:);
T_test=Ta(cv.test,:);
%%

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

% training and test set are put into a matrix together
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

%% PLS
[XL,yl,XS,YS,beta,PCTVAR] = plsregress(X_data', y_data', 11); % PLS regression

% fitted response is found here.
yfit = [ones(size(X_data',1),1) X_data']*beta;

% Fit for training
figure(1)
plotregression(y_data, yfit);
title('Regression Plot for Training Set','FontSize', 14);

% Prediction for training set
figure (2)
plot(y_data,'+');
hold on;
plot(yfit);
hold off;
title('Training set for PLS','FontSize', 14);
xlabel('Observation','FontSize', 12);
ylabel('Concentration','FontSize', 12); %(2*10^{-3}*mg/L)
legend({'Real','Predicted'})
xlim([10 30000])
ylim([-1.8 3])

% R^2 value is found here for the training.
stats=rs_stats(yfit, y_data')

%PLS components
figure(3)
subplot(1,2,1)
c = categorical(strseq('PC',1:11));
c = reordercats(c,strseq('PC',1:11));
bar(c,100*PCTVAR(2,:),0.5);
hold on
plot(c,cumsum(100*PCTVAR(2,:)), '-bo'); legend({'individual','cumulative'});
ylabel('% Explained variance in output','FontSize', 12); xlabel('Principal Components','FontSize', 12);
title('% Explained Variance in the Output','FontSize', 14);
hold off

% Residuals
subplot(1,2,2)
yfit_test = [ones(size(X_test,1),1) X_test]*beta;
residuals_test = y_test - yfit_test;
stem(residuals_test)
xlabel('Observation','FontSize', 12);
ylabel('Residual','FontSize', 12);
title('Residuals','FontSize', 14);
ylim([-7 4])
xlim([20 3500])

% Regression for test set
figure(4)
plotregression(y_test, yfit_test);
title('Regression Plot for Test Set','FontSize', 14);

%Fit for test
figure(5)
plot(y_test,'+');
hold on;
plot(yfit_test);
hold off;
xlabel('Observation','FontSize', 12);
ylabel('Concentration' ,'FontSize', 12); %(2*10^{-3}*mg/L)
legend({'Real','Predicted'})
title('Test Set for PLS','FontSize', 14);
xlim([10 3400])
ylim([-1.8 3])

% R^2 value is found here.
stats_test=rs_stats(yfit_test, y_test)
