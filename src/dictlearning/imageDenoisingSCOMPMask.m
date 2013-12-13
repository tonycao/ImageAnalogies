function [IOut, output] = imageDenoisingSCOMPMask(inputImg, inputMask, Dictionary, param)

[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
K                   = size(Dictionary,2);
lsregression        = 0;
if ~isfield(param, 'verbose')
    param.verbose = 0;
end

if ~isfield(param, 'L')
    param.L = 10;
end

% normalize dictionary
dnorm      = sqrt(sum(Dictionary(1:bb^2, :).^2, 1));
ind        = find(dnorm ~= 0);    % remove zeros
Dictionary(:,ind) = Dictionary(:,ind)./repmat(dnorm(ind), size(Dictionary,1), 1);


[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
[blocksmask,idxm] = my_im2col(inputMask,[bb,bb],slidingDis);
blocksError     = zeros(size(blocksraw));

alphas = zeros(K, size(blocksraw,2));
blocks = zeros(size(blocksraw));

param.numThreads = 4;
param.lambda2    = 0;
param.lambda     = 0;
%sparsity = zeros(1,size(blocksraw,2));

alphas = mexOMPMask(blocksraw(1:bb^2,:),Dictionary(1:bb^2,:),logical(abs(1-blocksmask)),param);
%alphas = mexLassoMask(blocksraw(1:bb^2,:),Dictionary(1:bb^2,:),logical(abs(1-blocksmask)),param);

sparsity = sum(abs(alphas)>1e-6,1)./K;

%for i = 1:size(blocksraw,2)
%    dictTemp = bsxfun(@times, (1-blocksmask(:,i)), Dictionary(1:bb^2,:));
%    alphas(:,i) = mexLasso(blocksraw(1:bb^2,i), dictTemp, param);
%    sparsity(i) = sum(abs(alphas(:,i))>1e-6)/K;
    % least squares regrssion based on selected dict items
%    if lsregression
%        ind = find(abs(alphas(:,i)) > 1e-3);
%        if ~isempty(ind)
%            seldict = dictTemp(:,ind);
%            alphas(ind,i) = seldict'*blocksraw(1:bb^2,i)\(seldict'*seldict);
%        end
%    end   
%    blocks(:,i) = dictTemp*alphas(:,i);
%end

totalpatch = sum(sum(abs(1-blocksmask),1)>0);
blocks = Dictionary(1:bb^2,:)*alphas.*abs(1-blocksmask);

overallSparsity = sum(sparsity)/size(blocksraw,2);
if param.verbose
    disp(['sparsity level ' num2str(overallSparsity)]);
end
%alphas = mexCD(blocksraw, Dictionary, sparse(size(Dictionary,2), size(blocksraw,2)),param);
%blocks        = Dictionary(1:bb^2,:)*alphas;
rectError     = sum((blocks-blocksraw).^2,1);
blocksError   = repmat(rectError,[size(blocksError,1) 1]);

output.alphas = alphas;
output.blocksError = blocksError;
output.blocksmask = blocksmask;
output.rectError = rectError;
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
