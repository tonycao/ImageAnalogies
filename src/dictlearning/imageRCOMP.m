function [IOut, output] = imageRCOMP(inputImg, Dictionary, param)

reduceDC            = param.reduceMean;
[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
maxBlocksToConsider = param.maxBlocksToConsider;
K                   = size(Dictionary,2);
L                   = param.L;
cvp                 = param.convexOptimization;
error               = 1e-3;

while (prod(floor((size(inputImg)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis        = slidingDis+1;
    output.slidingDis = slidingDis;
end
[blocks,idx]  = my_im2col(inputImg,[bb,bb],slidingDis);
%[blocks2]     = my_im2col(inputImg2,[bb,bb],slidingDis);
blocksPredict = zeros(size(blocks)); 
blocksError   = zeros(size(blocks));
predictError  = zeros(1,size(blocks,2));
% go with jumps of 30000
alphas        = zeros(K,size(blocks,2));
param.maxNumPatchesToTrainOn = 0;
for jj = 1:30000:size(blocks,2)
    jumpSize = min(jj+30000-1,size(blocks,2));
    
    if (reduceDC)
        vecOfMeans = mean(blocks(:,jj:jumpSize));
        blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    end
    
    % Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
    % Coefs = OMPerr(Dictionary(1:bb^2,:),blocks(:,jj:jumpSize),errT);
    param.maxNumPatchesToTrainOn = jumpSize-param.maxNumPatchesToTrainOn;
    %Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary(1:bb^2,:), param);
    
    if cvp == 0
        display('begin Orthogonal Matching Pursuit...');
        
        if L > 0 
            Coefs = OMP(Dictionary, blocks(:,jj:jumpSize), L);
        else
            Coefs = OMPerr(Dictionary, blocks(:,jj:jumpSize), 1e-3);
        end
    
    else
        display('begin Convex Optimization...');
        Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary, param);
    end
    
    alphas(:, jj:jumpSize) = Coefs;
    
    if (reduceDC)
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs + ones(size(blocks,1),1) * vecOfMeans;
    else
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs ;
        %blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs;

    end
end

%imin = min(blocksPredict(:));
%imax = max(blocksPredict(:));
%blocksPredict = (blocksPredict - imin)./(imax - imin);
%predictError = sum((blocksPredict-blocks2).^2,1);
%threshold = 28;
%blocksError = repmat(predictError,[size(blocksError,1) 1]);

%        predictError(jj:jumpSize) = sum((blocksPredict(:,jj:jumpSize)-blocks2(:,jj:jumpSize)).^2,1); 
%        blocksError(:,jj:jumpSize) = repmat(predictError(jj:jumpSize),[size(blocksError,1) 1]);

output.alphas = alphas;
%output.predictError = predictError;
%output.blocksError = blocksError;

count  = 1;
Weight = zeros(NN1,NN2);
%WeightPredict = zeros(size(Weight));
IMout  = zeros(NN1,NN2);
%IMoutPredict = zeros(size(IMout));
%IMoutError = zeros(size(IMout));
[rows,cols] = ind2sub(size(inputImg)-bb+1,idx);
for i  = 1:length(cols)
    col   = cols(i); row = rows(i);        
    block = reshape(blocks(:,count),[bb,bb]);
    %blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    %blockError = reshape(blocksError(:,count),[bb, bb]);
    
    IMout(row:row+bb-1,col:col+bb-1)  = IMout(row:row+bb-1,col:col+bb-1)+block;
    %IMoutPredict(row:row+bb-1,col:col+bb-1)=IMoutPredict(row:row+bb-1,col:col+bb-1)+blockPredict;
    %IMoutError(row:row+bb-1,col:col+bb-1) = IMoutError(row:row+bb-1,col:col+bb-1) + blockError;
    Weight(row:row+bb-1,col:col+bb-1) = Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

%IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
lambda           = 1;
IOut             = (lambda*inputImg+IMout)./(lambda*1+Weight);
%IOutPredict      = IMoutPredict./Weight;
%IoutError        = IMoutError ./ Weight;
%output.IoutError = IoutError;
output.Weight    = Weight;
%figure, imshow(IoutError, []);
%save('IoutError.mat', 'IoutError');
