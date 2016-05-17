% eigvector 特征向量
% eigvalue  特征值
% elapse    时间推移

function [eigvector, eigvalue, elapse] = PCA(data, options)
%% PCA	Principal Component Analysis
%
%	Usage:
%       [eigvector, eigvalue] = PCA(data, options)
%       [eigvector, eigvalue] = PCA(data)
% 
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%     options.ReducedDim   - The dimensionality of the reduced subspace. If 0,
%                         all the dimensions will be kept. 
%                         Default is 0. 
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem. 
%
%	Examples:
% 			fea = rand(7,10);
% 			[eigvector,eigvalue] = PCA(fea,4);
%           Y = fea*eigvector;
% 
%   version 2.2 --Feb/2009 
%   version 2.1 --June/2007 
%   version 2.0 --May/2007 
%   version 1.1 --Feb/2006 
%   version 1.0 --April/2004 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%                                                   

%%  判断是否有选项（降维数）
if (~exist('options','var'))
   options = [];
end

%% 给出降维数
ReducedDim = 0;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end

%% 判断降维数是否合理0<ReducedDim<样本维数
[nSmp,nFea] = size(data);
if (ReducedDim > nFea) || (ReducedDim <=0)
    ReducedDim = nFea; %如果大于维数活着小于等于0，就等于样本维数
end

%%  计时，可以用tic/toc更准确
tmp_T = cputime;

%% 如果稀疏矩阵的存储方式是sparse storage organization，则返回逻辑1；否则返回逻辑0
if issparse(data)
    data = full(data); %把稀疏矩阵转为全矩阵
end

%% 样本去均值化
sampleMean = mean(data,1); %求样本每一列均值 
data = (data - repmat(sampleMean,nSmp,1)); %去均值
 

%% 
if nFea/nSmp > 1.0713 % 维数／样本数
    % This is an efficient method which computes the eigvectors of
	% of A*A^T (instead of A^T*A) first, and then convert them back to
	% the eigenvectors of A^T*A.    
    ddata = data*data';
    ddata = max(ddata, ddata'); %获取每一个元素最大

    dimMatrix = size(ddata,2); %ddata的维数
    if dimMatrix > 1000 && ReducedDim < dimMatrix/10  % using eigs to speed up!
        option = struct('disp',0);
        %d = eigs(A,k,sigma,opts)  
        %返回k个最大特征值,
        %sigma取值：'lm' 表示绝对值最大的特征值；'sm' 绝对值最小特征值；
        %对实对称问题：'la'表示最大特征值；'sa'为最小特征值；
        %对非对称和复数问题：'lr' 表示最大实部；'sr' 表示最小实部；
        %'li' 表示最大虚部；'si'表示最小虚部
        [eigvector, eigvalue] = eigs(ddata,ReducedDim,'la',option);%求特征向量 特征值
        eigvalue = diag(eigvalue);%创建对角矩阵
    else
        [eigvector, eigvalue] = eig(ddata); %直接算特征向量特征值
        eigvalue = diag(eigvalue); %对角阵

        [junk, index] = sort(-eigvalue); %junk排序好的向量index是索引-为了从大到小
        eigvalue = eigvalue(index);
        eigvector = eigvector(:, index); %这两步重新排序
    end

    clear ddata;
    
    maxEigValue = max(abs(eigvalue)); %abs绝对值，求最大特征值
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12); %获取很小的特征值
    eigvalue (eigIdx) = []; %将对应的特征值置空
    eigvector (:,eigIdx) = [];%将对应的特征向量置空

    % 不懂
    eigvector = data'*eigvector;		% Eigenvectors of A^T*A
	eigvector = eigvector*diag(1./(sum(eigvector.^2).^0.5)); % Normalization
else
    ddata = data'*data;
    ddata = max(ddata, ddata');

    dimMatrix = size(ddata,2);
    if dimMatrix > 1000 & ReducedDim < dimMatrix/10  % using eigs to speed up!
        option = struct('disp',0);
        [eigvector, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        [eigvector, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);

        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        eigvector = eigvector(:, index);
    end
    
    clear ddata;
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
    eigvalue (eigIdx) = [];
    eigvector (:,eigIdx) = [];
end

%% 
if ReducedDim < length(eigvalue)
    eigvalue = eigvalue(1:ReducedDim);
    eigvector = eigvector(:, 1:ReducedDim);
end

%% 
if isfield(options,'PCARatio') %PCA比
    sumEig = sum(eigvalue);
    sumEig = sumEig*options.PCARatio; %特征值和X比例
    sumNow = 0;
    for idx = 1:length(eigvalue)
        sumNow = sumNow + eigvalue(idx);
        if sumNow >= sumEig
            break;
        end
    end
    eigvector = eigvector(:,1:idx);
end
%% 计算花费的时间
elapse = cputime - tmp_T;