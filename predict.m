%% 预处理数据
clear
clc
tic; %计时器开始
preprocess

%% 得到每个分类器对每个样本的预测 
[ svmStructs ] = multi( trainX,trainY ); %多分类调用得到svmStruct的数组
predicted = zeros(size(testY)); %先定义空的预测
types = unique(testY); %获取类型
predictMatrix = zeros(size(testY,1),size(types,1)); %定义预测矩阵
predictMatrix(:,end)=1; %最后一列为1，因为假如前面都不是，那就是最后一类
for i = 1:size(svmStructs,1) %每一个分类器分类
    svm_struct  = svmStructs{i};
    Group = svmclassify(svm_struct,testX);
    predictMatrix(:,i) = Group;
end
clear Group i svm_struct;

%% 获取到第一个1出现的位置，就是分类的类别
postion = zeros(size(testY)); 
for i=1:size(predictMatrix,1)
    temp = predictMatrix(i,:);
    one=find(temp==1);
    postion(i,1) = one(1);
end
clear i temp one;

%% 将位置转化为类别
predicted = types(postion); 
clear i temp postion;

%%
err = sum(predicted ~= testY);
Accuracy = 1-err/size(testY,1);
Rightrating =strcat(num2str(Accuracy*100),'%');
toc;
fprintf('正确率为%s\n', Rightrating);



