function [blocksPredict, output] = patchReconstructSC(blockscomb, Dictionary, param)

% image Analogy with Spare Representation model
%

reduceDC            = param.reduceMean;
K                   = size(Dictionary,2);
psl                 = param.patchSideLength;

%blockscomb          = [blocksraw; blocks2];
blocks              = zeros(size(blockscomb));
blocksPredict       = zeros(size(blockscomb));
predictBlockError   = zeros(1, size(blockscomb,2));
%predictBlockResidual= zeros(size(blockscomb));

% go with jumps of 30000
alphas = zeros(K,size(blockscomb,2));
param.maxNumPatchesToTrainOn = 0;
round = 0;
for jj = 1:30000:size(blockscomb,2)
    round = round + 1;
    display(['begin sparse coding... round = ' num2str(round)]);
    jumpSize = min(jj+30000-1,size(blockscomb,2));
    if (reduceDC)
        vecOfMeans = mean(blockscomb(:,jj:jumpSize));
        blocks(:,jj:jumpSize) = blockscomb(:,jj:jumpSize) - repmat(vecOfMeans,size(blockscomb,1),1);
    else
        blocks(:,jj:jumpSize) = blockscomb(:,jj:jumpSize);
    end
    
    % Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
    % Coefs = OMPerr(Dictionary(1:bb^2,:),blocks(:,jj:jumpSize),errT);
    param.maxNumPatchesToTrainOn = jumpSize-param.maxNumPatchesToTrainOn;
    
    % compute coefficient using sparse coding
    Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary, param);
    alphas(:, jj:jumpSize) = Coefs;
    
    
    if (reduceDC)
        blocksPredict(:,jj:jumpSize) = Dictionary*Coefs + ones(size(blockscomb,1),1) * vecOfMeans;
    else
        % blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs;
        blocksPredict(:,jj:jumpSize) = Dictionary*Coefs;
    end
    
    predictBlockError(jj:jumpSize) = sum(abs(blocksPredict(:,jj:jumpSize)-blocks(:,jj:jumpSize)), 1);
end

%imin          = min(blocksPredict(:));
%imax          = max(blocksPredict(:));
%blocksPredict = (blocksPredict - imin)./(imax - imin);
predictError  = blocksPredict - blockscomb;
%blocksError   = repmat(predictError,[size(blocksError,1) 1]);

output.alphas       = alphas;
output.predictError = predictError;
%output.blocksError  = blocksError;
output.predictBlockError = predictBlockError;
