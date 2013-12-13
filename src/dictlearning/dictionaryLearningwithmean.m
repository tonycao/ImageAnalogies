function [Dictionary, output, vecMeans, rawPatch] = dictionaryLearningwithmean(image, param)

%==========================================================================
% Dictionary Learning Code
%
% by Tian Cao
%==========================================================================

reduceMean = 1;% param.reduceMean;
psl = param.patchSideLength;
K   = param.dictSize;
trainingSize = param.trainingSize;
maxNumBlocksToTrainOn = fix(param.maxNumBlocksToTrainOn/trainingSize);

%if trainingSize*maxNumBlocksToTrainOn >
%patchMatrix = zeros(psl^2*param.numberofImages, maxNumBlocksToTrainOn*trainingSize);

iPatch = 0;

for ind = 1:trainingSize
    %[imSize1, imSize2] = size(image(:,:,1));
    [imSize1, imSize2] = size(image{ind*2});
    
    maxPatchNum = prod([imSize1, imSize2]-psl+1);
    
    %waitBarOn = 1;
    img1 = image{(ind-1)*2+1};
    img2 = image{(ind-1)*2+2};
    
    % train a dictionary on blocks from the noisy image
    
    if(prod([imSize1, imSize2]-psl+1) > maxNumBlocksToTrainOn)
        randPermutation = randperm(prod([imSize1, imSize2]-psl+1));
        selectedPatches = randPermutation(1:maxNumBlocksToTrainOn);
        
        %patchMatrix = zeros(psl^2, maxNumPatchesToTrainOn);
        for i = 1:maxNumBlocksToTrainOn
            iPatch = iPatch + 1;
            [row,col] = ind2sub([imSize1, imSize2]-psl+1, selectedPatches(i));
            currPatch1 = img1(row:row+psl-1, col:col+psl-1);
            %patchMatrix(1:psl^2,iPatch) = currPatch(:);
            currPatch2 = img2(row:row+psl-1, col:col+psl-1);
            patchMatrix(:,iPatch) = [currPatch1(:);currPatch2(:)];
        end
        
    else
        
        iPatch = iPatch + 1;%prod([imSize1, imSize2]-psl+1);
        patchMatrix1 = im2col(img1, [psl, psl], 'sliding');
        patchMatrix2 = im2col(img2, [psl, psl], 'sliding');
        patchMatrixSize = size(patchMatrix1,2);
        %patchMatrix = zeros(psl^2*param.numberofImages, patchMatrixSize);
        patchMatrix(:, iPatch:iPatch+patchMatrixSize-1) = ...
            [patchMatrix1; patchMatrix2];
        iPatch = iPatch+patchMatrixSize-1;
        %patchMatrix = patchMatrix1;
    end
end

rawPatch = patchMatrix(1:psl*psl,:);

if(reduceMean)
    vecMeans1 = mean(patchMatrix(1:psl*psl,:));
    vecMeans2 = mean(patchMatrix(psl*psl+1:end,:));
    vecMeans  = [vecMeans1; vecMeans2];
    patchMatrix = patchMatrix-[ones(psl*psl,1)*vecMeans1; ones(psl*psl,1)*vecMeans2];
    patchMatrixwithMean = [patchMatrix(1:psl^2, :); vecMeans1; ...
        patchMatrix(1+psl^2:end, :); vecMeans2];
end


param.displayProgress = 1;
%param.dictSize = param.dictSize+2;
[Dictionary, output] = doDictLearningwithMean(patchMatrixwithMean, param);


