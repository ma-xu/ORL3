function [w] = LDA(LX,d,nv)
%  LX:样本； d:标签； nv：前nv个特征向量

[m,n]=size(LX);
M=mean(LX); %列均值

Sb=sparse(zeros(n,n)); %n*n的0稀疏矩阵
Sw=sparse(zeros(n,n));

for i=unique(d') %提取所有类别标签
    Xc=LX(d==i,:); %当前标签下的样本
    [m1,n1]=size(Xc); 
    mec=mean(Xc); %类内均值
    Sw=Sw+(Xc-ones(m1,1)*mec)'*(Xc-ones(m1,1)*mec); %加上类内方差
    Sb=Sb+m1*(mec-M)'*(mec-M); %加上总内间方差
end

St=Sw+Sb; 
[U,V]=eig(Sb,St);
B=diag(V);
[B,index]=sort(B,'descend');

w=U(:,index(1:nv,1));
% w=orth(w);