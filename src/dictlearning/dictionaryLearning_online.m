function [Dictionary] = dictionaryLearning_online(image, param)

%==========================================================================
% online Dictionary Learning Code
%
% by Tian Cao
%==========================================================================


if ~isfield(param, 'rawpatchDict')
    param.rawpatchDict = 0;
end 
reduceMean = param.reduceMean;
psl        = param.patchSideLength;
trainingSize          = size(image,2)/2;
if ~isfield(param, 'dictSize')
    dictSize = param.patchSideLength^2*param.RR;
else
    dictSize = param.dictSize;
end

fprintf('Generating training patches...\n');

patchMatrix = [];

for ind = 1:trainingSize
    
    img1 = image{(ind-1)*2+1};
    img2 = image{(ind-1)*2+2};
    
    patchMatrix1 = my_im2col(img1, [psl, psl], param.slidingDis);
    patchMatrix2 = my_im2col(img2, [psl, psl], param.slidingDis);
    
    patchMatrixtemp = [patchMatrix1; patchMatrix2];
    patchMatrix     = [patchMatrix patchMatrixtemp];
    
end

if param.rawpatchDict
    patchNum = size(patchMatrix,2);
    sel = randperm(patchNum);
    Dictionary = patchMatrix(:,sel(1:dictSize));
    dnorm      = repmat(sqrt(sum(Dictionary.^2)),[size(Dictionary,1) 1]);
    ind        = (dnorm == 0);
    dnorm(ind) = 1e-6;
    Dictionary = Dictionary./dnorm;
    
else
    
    npatch = size(patchMatrix,2);
    randpatch = randperm(npatch);
    patchMatrix = patchMatrix(:,randpatch);
    
    if(reduceMean)
        vecMeans1   = mean(patchMatrix(1:psl*psl,:));
        vecMeans2   = mean(patchMatrix(psl*psl+1:end,:));
        vecMeans    = [vecMeans1; vecMeans2];
        patchMatrix = patchMatrix-[ones(psl*psl,1)*vecMeans1; ones(psl*psl,1)*vecMeans2];
    end
    
    %param.dictSize = size(patchMatrix)
    %param.displayProgress = 1;
    %[Dictionary, output]  = doDictLearning(patchMatrix, param);
    
    % onlien dictionary learning
    dnorm      = repmat(sqrt(sum(patchMatrix.^2)),[size(patchMatrix,1) 1]);
    ind        = (dnorm == 0);
    dnorm(ind) = 1e-6;
    
    param.K          = dictSize;
    param.numThreads = 2; % number of threads
    param.batchsize  = 500;
    param.mode       = 2;
    param.iter       = 1000;
    param.verbose    = 1;
    for i = 1:size(patchMatrix,2)
        patchMatrix(:,i) = patchMatrix(:,i)./dnorm(:,i);
    end
    %Dictionary = mexTrainDL_Memory(patchMatrix./dnorm, param);
    Dictionary = mexTrainDL(patchMatrix, param);
    %Dictionary = doDictLearning(patchMatrix, param);
end

