function [matDictionary,alphas] = doDictLearningwithMean(vecPatches, param)

%==========================================================================
% description
%
%
% by Tian Cao
%==========================================================================

% display initial dictionary
displayinitDict = 1;

% extract patches as 1D vectors
psl = param.patchSideLength;
patchSize = psl^2;
dictSize = param.dictSize;
maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
patchNum = size(vecPatches,2);

%% initialize matrix
%alphas = zeros(dictSize, maxNumBlocksToTrainOn);
alphas = zeros(dictSize, patchNum);

%sel = randperm(maxNumBlocksToTrainOn);
sel = randperm(patchNum);
sel = sel(1:dictSize);
matDictionary = vecPatches(:,sel);

if displayinitDict == 1
    figure;
    K = param.dictSize;
    bb = param.patchSideLength;
    %subplot(1,2,1);
    %     I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
    %     floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
    
    %subplot(1,2,1);
    %I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
    %    floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb,0);
    %subplot(1,2,2);
    %I = displayDictionaryElementsAsImage(matDictionary(1+bb^2:end,:), floor(sqrt(K)), ...
    %    floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb,0);
end

if param.rawpatchDict ~= 1
    % normalize patches
    matDictNorm = sqrt(sum(matDictionary(1:psl^2,:).^2));
    
    ind = find(matDictNorm ~= 0);
    
    matDictionary(:,ind) = matDictionary(:,ind)./repmat(matDictNorm(1,ind),[size(vecPatches,1),1]);
    matDictionary(find(isnan(matDictionary))) = 0;
    
    %dNorm = sqrt(diag(matDictionary'*matDictionary));
    %matDictionary = matDictionary./repmat(dNorm',[patchSize*param.numberofImages,1]);
    %matDictionary = matDictionary./repmat(sqrt(sum(matDictionary.^2)),[patchSize,1]);
    
    %% begin iteration
    for iter = 0:param.maxNumberofDictionaryUpdate-1
        
        %progressbar(iter,param.maxNumberofDictionaryUpdate-1);
        % save intermediate dictionary output if desired
        if ( param.saveIntermediateDictionaryResults )
            
            %save the intermediate dictionary results (so we can follow the progress of the optimization)
            outputFileNamePrefix = sprintf('iter-%05d-dictionaryPatchImage',iter);
            SaveCurrentDictionaryAsImage( outputFileNamePrefix, matDictionary, param );
        end
        
        tic
        
        % compute coefficients using the current dictionary
        fprintf('\nsparse coding...\n\n');
        alphas = sparsecoding(vecPatches, matDictionary, param);
        %oldalphas = alphas;
        
        fprintf('\nupdate dictionary...\n\n');
        % update the dictionary
        matDictionary = updatedictionarywithMean(vecPatches, matDictionary, alphas, param);
        matDictionary(find(isnan(matDictionary))) = 0;
        %matDictionary(1:patchSize,:) = matDictionary(1:patchSize,:)./repmat(sqrt(sum(matDictionary(1:patchSize,:).^2)), patchSize,1);
        %oldmatDictionary = matDictionary;
        
        [currentEnergy, l2Energy, l1Energy] = computecurrentenergy(vecPatches, matDictionary, alphas, param);
        fprintf( 'Iter %d complete, Energy = %6.4d\n, l2Energy = %6.4d\n, l1Energy = %6.4d\n', ...
            iter+1,  currentEnergy, l2Energy, l1Energy);
        
        toc
        
        fprintf('\n');
        
%         figure;
%         K = param.dictSize;
%         bb = param.patchSideLength;
%         subplot(1,2,1);
%         I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
%             floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
%         subplot(1,2,2);
%         I = displayDictionaryElementsAsImage(matDictionary(1+bb^2:end,:), floor(sqrt(K)), ...
%             floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
    end
end
%figure;
%K = param.dictSize;
%bb = param.patchSideLength;
%subplot(1,2,1);
%I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
%   floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
%subplot(1,2,2);
%I = displayDictionaryElementsAsImage(matDictionary(1+bb^2:end,:), floor(sqrt(K)), ...
%   floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
%
% save result
%outputFileNamePrefix = 'result-dictionaryPatchImage';
%savecurrentdictionaryasimage(outputFileNamePrefix, matDictionary, param);

function [currentEnergy, l2Energy, l1Energy] = computecurrentenergy(vecPatches, matDictionary, alphas, param)

vecMisMatch = vecPatches - matDictionary*alphas;
%currentEnergy = 0.5*sum(vecMisMatch(:).^2) + param.lambda*sum(abs(alphas(:)));
currentEnergy = 0.5*norm(vecMisMatch(:),2)^2 + param.lambda*norm(alphas(:), 1);
l2Energy = 0.5*norm(vecMisMatch(:),2)^2;
l1Energy = param.lambda*norm(alphas(:), 1);
