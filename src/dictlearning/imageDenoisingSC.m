function [IOut, output] = imageDenoisingSC(inputImg, Dictionary, param)

[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
K = size(Dictionary,2);

% normalize dictionary
dnorm      = sqrt(sum(Dictionary(1:bb^2, :).^2, 1));
ind        = find(dnorm ~= 0);    % remove zeros
Dictionary(:,ind) = Dictionary(:,ind)./repmat(dnorm(ind), size(Dictionary,1), 1);


[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
blocksError     = zeros(size(blocksraw));

param.numThreads = 4;

alphas = mexLasso(blocksraw, Dictionary(1:bb^2,:), param);
%alphas = mexCD(blocksraw, Dictionary, sparse(size(Dictionary,2), size(blocksraw,2)),param);
blocks        = Dictionary(1:bb^2,:)*alphas;
rectError     = sum((blocks-blocksraw).^2,1);
blocksError   = repmat(rectError,[size(blocksError,1) 1]);

output.alphas = alphas;
output.blocksError = blocksError;
count  = 1;
Weight = zeros(NN1,NN2);
IMout  = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(inputImg)-bb+1,idx);

for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    block =reshape(blocks(:,count),[bb,bb]);
    %blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    %IMoutPredict(row:row+bb-1,col:col+bb-1)=IMoutPredict(row:row+bb-1,col:col+bb-1)+blockPredict;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    %WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

%IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
IOut = (IMout)./(Weight);
%IOutPredict = IMoutPredict./WeightPredict;
