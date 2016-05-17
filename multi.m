function [ svmStructs ] = multi( trainX,trainY )
    types = unique(trainY);%定义总样本类别数
    svmStructs = cell(size(types,1)-1,1);%定义细胞分类器
    for i=1:size(types,1)-1
        labels = -ones(size(trainY));%-1,1两类
        I = find(trainY==types(i,1));
        labels(I) = 1;
        svmStruct=svmtrain(trainX,labels);%自带svm训练
        svmStructs{i} = svmStruct;
    end
end

