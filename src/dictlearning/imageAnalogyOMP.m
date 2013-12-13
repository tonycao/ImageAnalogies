function [IOut, IOutPredict, output] = imageAnalogyOMP(inputImg, inputImg2, Dictionary, param)

% image Analogy with Spare Representation model
%
%
if ~isfield(param, 'reduceMean')
    param.reduceMean = 0;
end
reduceDC            = param.reduceMean;
[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
%maxBlocksToConsider = param.maxBlocksToConsider;
%maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
K = size(Dictionary,2);

% normalize dictionary
dnorm      = sqrt(sum(Dictionary(1:bb^2, :).^2, 1));
ind        = find(dnorm ~= 0);    % remove zeros
Dictionary(:,ind) = Dictionary(:,ind)./repmat(dnorm(ind), size(Dictionary,1), 1);

[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
[blocks2,idx2]  = my_im2col(inputImg2,[bb,bb],slidingDis);
blocks          = zeros(size(blocksraw));
blocksPredict   = zeros(size(blocksraw));
blocksError     = zeros(size(blocksraw));
predictError    = zeros(1,size(blocksraw,2));


% batch processing with 30000
alphas = zeros(K,size(blocksraw,2));
%param.maxNumPatchesToTrainOn = 0;

param.numThreads = 4;
    %tic
    alphas = mexOMP(blocksraw, Dictionary(1:bb^2,:), param);
    %toc

blocksPredict = Dictionary(1+bb^2:end,:)*alphas;
blocks        = Dictionary(1:bb^2,:)*alphas;

predictError  = sum((blocksPredict-blocks2).^2,1);
blocksError   = repmat(predictError,[size(blocksError,1) 1]);

%        predictError(jj:jumpSize) = sum((blocksPredict(:,jj:jumpSize)-blocks2(:,jj:jumpSize)).^2,1);
%        blocksError(:,jj:jumpSize) = repmat(predictError(jj:jumpSize),[size(blocksError,1) 1]);

output.alphas       = alphas;
output.predictError = predictError;
output.blocksError  = blocksError;
output.inputImg2    = inputImg2;

count  = 1;
Weight = zeros(NN1,NN2);
%WeightPredict = zeros(size(Weight));
IMout  = zeros(NN1,NN2);
IMoutPredict = zeros(size(IMout));
IMoutError   = zeros(size(IMout));
[rows,cols]  = ind2sub(size(inputImg)-bb+1,idx);

% reconstruct image
for i  = 1:length(cols)
    col   = cols(i);
    row   = rows(i);
    block = reshape(blocks(:,count),[bb,bb]);
    blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    blockError   = reshape(blocksError(:,count),[bb, bb]);
    
    IMout(row:row+bb-1,col:col+bb-1)        = IMout(row:row+bb-1,col:col+bb-1)+block;
    IMoutPredict(row:row+bb-1,col:col+bb-1) = IMoutPredict(row:row+bb-1,col:col+bb-1) + blockPredict;
    IMoutError(row:row+bb-1,col:col+bb-1)   = IMoutError(row:row+bb-1,col:col+bb-1) + blockError;
    Weight(row:row+bb-1,col:col+bb-1)       = Weight(row:row+bb-1,col:col+bb-1) + 1;%ones(bb);
    %WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1) + 1;%ones(bb);
    count = count+1;
end;

%IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
lambda = 0.02;
%IOut   = (lambda*inputImg+IMout)./(lambda*1+Weight);
IOut   = IMout./Weight;
IOutPredict = IMoutPredict./Weight;
IoutError   = IMoutError ./ Weight;
output.IoutError = IoutError;
output.Weight    = Weight;
%figure, imshow(IoutError, []);
%save('IoutError.mat', 'IoutError');
