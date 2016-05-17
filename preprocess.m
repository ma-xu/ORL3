%先通过PCA提取再通过LDA降维，防止Sw奇异
% 正确率100%
clear
clc
load('Yale_32x32.mat');
% 选取测试集，每一类的最后一个
testNum = [11,22,33,44,55,66,77,88,99,110,121,132,143,154,165];

%注释掉绘画人脸部分
%{
%画出待测试的人脸
testPeople = fea(testNum,:);
for i = 1:size(testNum,2)
    people = reshape(testPeople(i,:),32,32);
    subplot(3,5,i);
    imshow(people/256);
    hold on;
end
%}

trainY = gnd;
trainX = fea;

%PCA 先提取
options.PCARatio=0.95;
[eigvector,eigvalue] = PCA(trainX,options);
trainX = trainX*eigvector;
clear eigvector  eigvalue options;

%LDA获取降维后的数据
[w] = LDA(trainX,trainY,13);
trainX = trainX*w;
clear w;


%拆分训练集和测试集
testX = trainX(testNum,:);
testY = trainY(testNum,:);
trainX(testNum,:) = [];
trainY(testNum,:) = [];
clear fea;
clear gnd;