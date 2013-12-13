function [matDictionary] = updatedictionary(vecPatches, matDictionary, alphas, param)

%==========================================================================
% description
%
%
% by Tian Cao
%==========================================================================

%% solves D = (\sum_{i=1}^N p_i\alpha_i^T)(mu I +\sum_{i=1}^N\alpha_i \alpha_i^T)^{-1}
% initialize matrix
patchSize = param.patchSideLength^2;

if ~isfield(param, 'display')
    display = 0;
else
    display = param.display;
end
% normalize the dictionary
%dictionary = matDictionary;
%matDictionary(1:patchSize,:) = matDictionary(1:patchSize,:)./repmat(sqrt(sum(matDictionary(1:patchSize,:).^2)), patchSize,1);
% dictionary(1+patchSize:end,:) = matDictionary(1+patchSize:end,:)./repmat(sqrt(sum(dictionary(1+patchSize:end,:).^2)), patchSize,1);

% initalize MuI
muI = eye(param.dictSize);

muI = muI.*param.mu;

%tic
m1 = vecPatches*alphas';
m2 = alphas*alphas';
%toc

% invert M2 and pre-multiply by M1 to obtain the new estimate of the
% dictionary
%dictionary = m1*inv(muI+m2);
matDictionary = m1*inv(muI+m2);
%matDictionary = m1*pinv(m2);

% display
if display ==1
    figure;
    K  = param.dictSize;
    bb = param.patchSideLength;
    %subplot(1,2,1);
    displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
        floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
end

matDictionary = matDictionary./repmat(sqrt(sum(matDictionary.^2)), size(matDictionary,1),1);
matDictionary(find(isnan(matDictionary))) = 0;

%dNorm = sqrt(diag(matDictionary'*matDictionary));
%matDictionary = matDictionary./repmat(dNorm', patchSize*param.numberofImages,1);
%matDictionary = matDictionary./repmat(sqrt(sum(matDictionary.^2)), patchSize,1);

%display
if display
    figure;
    K  = param.dictSize;
    bb = param.patchSideLength;
    %subplot(1,2,1);
    displayDictionaryElementsAsImage(matDictionary(1:bb^2,:), floor(sqrt(K)), ...
        floor(size(matDictionary,2)/floor(sqrt(K))),bb,bb);
end


%% normlize all the columns for each image patch part
%sumSqr = zeros(param.numberofImages, param.numberofDictionaryElements);
%norm = zeros(param.numberofImages, param.numberofDictionaryElements);

%sumSqr(1,:) = sum(dictionary(1:patchSize,:).^2, 1);%./repmat(patchSize, 1, param.numberofDictionaryElements);
%sumSqr(2,:) = sum(dictionary(patchSize+1:end,:).^2, 1);% ./repmat(patchSize, 1, param.numberofDictionaryElements);

%norm = sqrt(sum(dictionary(1:patchSize,:).^2));
%norm(2,:) = sqrt(sumSqr(2,:));

%indzeros = find(norm<=0);

%norm(indzeros) = ones(size(indzeros));

%dictionary(1:patchSize,:) = dictionary(1:patchSize,:)./repmat(norm(1,:), ...
%                                patchSize, 1);
% dictionary(patchSize+1:end,:) = dictionary(patchSize+1:end,:)./repmat(norm(2,:), ...
%                                 patchSize, 1);
%
%matDictionary = dictionary;
%%
%matDictionary;

