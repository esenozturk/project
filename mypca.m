%%
clear;clc;
load onlinedata  % Import Data

%Inputs and outputs are defined here.
inputs = @(T) [T.input1 T.input2 T.input3 T.input4 T.input5 T.input6 T.input7 T.input8 T.input9 T.input10 T.input11]; % 6 columns
output = @(T) [T.output];
C31s = struct;

% 10% of the dataset is separated for test
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

%training and test set are put into a matrix together
X_total = [X_data; X_test];
y_total = [y_data; y_test];


% PCA 
Z=zscore(X_total);
[coef, score, a, b, explained] = pca(Z);
coeff = pca(Z)
explained

% PC components
figure(2)
subplot(1,2,1)
c = categorical(strseq('PC',1:11));
c = reordercats(c,strseq('PC',1:11));
bar(c,explained,0.5); 
hold on
cum_explained = cumsum(explained);
plot(c,cum_explained, '-bo'); legend({'individual','cumulative'});
ylabel('% Explained variance','FontSize', 14); xlabel('Principle Components','FontSize', 14);
title('% Explained Variance in Input','FontSize', 14)

%Biplot
subplot(1,2,2)
[coeff,score,latent] = pca(Z);
Xcentered = score*coeff';
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',{'Tank B','Tank C','pH','O2%','CO2 tot','CO2 flow','O2 tot','O2 flow','Agitation','Temperature','Pressure'});
title('PCA Biplot','FontSize', 14)
