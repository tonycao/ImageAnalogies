function [Dictionary, output, rawPatch, vecMeans] = dictionaryLearning(image, param)

%==========================================================================
% Dictionary Learning Code
%
% by Tian Cao
%==========================================================================

if ~isfield(param, 'reduceMean')
    param.reduceMean = 0;
end
if ~isfield(param, 'dictSize')
    dictSize = param.patchSideLength^2*param.RR;
else
    dictSize = param.dictSize;
end
reduceMean = param.reduceMean;
psl        = param.patchSideLength;
%K          = param.dictSize;
%trainingSize          = param.trainingSize;
trainingSize = size(image,2)/2;
%maxNumBlocksToTrainOn = fix(param.maxNumBlocksToTrainOn/trainingSize);

%if trainingSize*maxNumBlocksToTrainOn >
%patchMatrix = zeros(psl^2*param.numberofImages, maxNumBlocksToTrainOn*trainingSize);

iPatch = 0;

fprintf('Generating training patches...\n');

patchMatrix = [];

for ind = 1:trainingSize
    
    img1 = image{(ind-1)*2+1};
    img2 = image{(ind-1)*2+2};
    %mask1 = mask1 == 0;
    %mask1 = sum(mask1);
    %ind = find(mask>0);
    patchMatrix1 = my_im2col(img1, [psl, psl], param.slidingDis);
    patchMatrix2 = my_im2col(img2, [psl, psl], param.slidingDis);
    %patchmask = my_im2col(mask1, [psl, psl], param.slidingDis);
    %patchmask = sum(patchmask);
    %indmask = find(patchmask > 0);
    
    %patchMatrix1(:, indmask) = [];
    %patchMatrix2(:, indmask) = [];
    
    patchMatrixtemp = [patchMatrix1; patchMatrix2];
    patchMatrix     = [patchMatrix patchMatrixtemp];
    
end

%if ~param.normalized
%    threshold = 4;
%else
%    threshold = 0.01;
%    threshold = 0.001;
%end

% patch pruning
%patchMatrix = patchPruning(patchMatrix(1:psl^2,:), patchMatrix, threshold);
%patchMatrix = patchPruning(patchMatrix(1+psl^2:end,:), patchMatrix, threshold);

%if size(patchMatrix,2) > param.maxBlocksToConsider
%    ind = randperm(size(patchMatrix,2),param.maxBlocksToConsider);
%    patchMatrix = patchMatrix(:,ind);
%end



%rawPatch = patchMatrix(1:psl*psl,:);

%[coefs, scores] = princomp([patchMatrix']);
%gscatter(scores(:,1), scores(:,2), [], [], '+*');

if(reduceMean)
    vecMeans1   = mean(patchMatrix(1:psl*psl,:));
    vecMeans2   = mean(patchMatrix(psl*psl+1:end,:));
    vecMeans    = [vecMeans1; vecMeans2];
    patchMatrix = patchMatrix-[ones(psl*psl,1)*vecMeans1; ones(psl*psl,1)*vecMeans2];
end

%param.dictSize = size(patchMatrix)
param.displayProgress = 1;
%[Dictionary, output]  = doDictLearning(patchMatrix, param);

% onlien dictionary learning
output     = [];
dnorm      = repmat(sqrt(sum(patchMatrix.^2)),[size(patchMatrix,1) 1]);

param.K          = dictSize;
param.numThreads = 4; % number of threads
param.batchsize  = 800;
param.mode       = 2;
param.iter       = 800;
param.verbose    = 1;

disp('begin dictionary learning...');
kmeans = 0;
if kmeans
    tic
    %[~,Dictionary,~] = fkmeans((patchMatrix./dnorm)', dictSize);    
    Dictionary=pca((patchMatrix./dnorm)');
    toc
    %Dictionary = Dictionary';
else
Dictionary = mexTrainDL(patchMatrix./dnorm, param);
end

%for i = 1:size(patchMatrix,2)
%    patchMatrix(:,i) = patchMatrix(:,i)./dnorm(:,i);
%end
%Dictionary = mexTrainDL_Memory(patchMatrix./dnorm, param);
%Dictionary = mexTrainDL(patchMatrix, param);
