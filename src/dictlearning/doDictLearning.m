function [matDictionary,alphas] = doDictLearning(vecPatches, param)

%==========================================================================
% description
%
%
% by Tian Cao
%==========================================================================

% display initial dictionary
if ~isfield(param, 'displayinitDict')
    displayinitDict = 0;
else
    displayinitDict = param.displayinitDict;
end
if ~isfield(param, 'verbose')
    verbose = 0;
else
    verbose = param.verbose;
end

if verbose
    fprintf('Begin dictionary learning...\n');
end

% extract patches as 1D vectors
psl       = param.patchSideLength;
patchSize = psl^2;
dictSize  = param.dictSize;
maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
%patchNum  = size(vecPatches,2);

%% initialize matrix
alphas = zeros(dictSize, maxNumBlocksToTrainOn);
%alphas    = zeros(dictSize, patchNum);

sel = randperm(maxNumBlocksToTrainOn);
vecPatches = vecPatches(:,sel);
sel       = sel(1:dictSize);
matDictionary = vecPatches(:,sel);

if displayinitDict
    figure;
    K = param.dictSize;
    bb = param.patchSideLength;
    %subplot(1,2,1);
    %     I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
    %     floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
    
    subplot(1,2,1);
    I = displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
       floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb,0);
    subplot(1,2,2);
    I = displayDictionaryElementsAsImage(matDictionary(1+bb^2:end,:), floor(sqrt(K)), ...
       floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb,0);
end

	% normalize patches
	matDictNorm = sqrt(sum(matDictionary.^2,1)); %
    %matDictNorm = sqrt(sum(matDictionary(1:patchSize,:).^2)); %
    
    ind = find(matDictNorm ~= 0);
    
    matDictionary(:,ind) = matDictionary(:,ind)./repmat(matDictNorm(1,ind),[patchSize*param.numberofImages,1]);
    matDictionary(find(isnan(matDictionary))) = 0;

if param.rawpatchDict ~= 1
    % normalize patches

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
        if verbose
            fprintf('\nSparse coding...\n\n');
        end
        tic
        alphas = sparsecoding(vecPatches, matDictionary, param);
        %oldalphas = alphas;
        toc
        if verbose
            fprintf('\nUpdate dictionary...\n\n');
        end
        % update the dictionary
        matDictionary = updatedictionary(vecPatches, matDictionary, alphas, param);
        matDictionary(find(isnan(matDictionary))) = 0;
        %matDictionary(1:patchSize,:) = matDictionary(1:patchSize,:)./repmat(sqrt(sum(matDictionary(1:patchSize,:).^2)), patchSize,1);
        %oldmatDictionary = matDictionary;
        
        [currentEnergy, l2Energy, l1Energy] = computecurrentenergy(vecPatches, matDictionary, alphas, param);
        if verbose
            fprintf( 'Iter %d complete, Energy = %6.4d\n, l2Energy = %6.4d\n, l1Energy = %6.4d\n', ...
                iter+1,  currentEnergy, l2Energy, l1Energy);
            fprintf('\n');
        end
        toc
       
        
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
