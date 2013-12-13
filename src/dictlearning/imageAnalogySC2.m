function [IOut, IOutPredict, output] = imageAnalogySC2(inputImg, inputImg2, Dictionary, param)

% image Analogy with Spare Representation model
%
%
if ~isfield(param, 'reduceMean')
    param.reduceMean = 0;
end

if ~isfield(param, 'lasso')
    param.lasso = 1;
end
reduceDC            = param.reduceMean;
[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
%maxBlocksToConsider = param.maxBlocksToConsider;
%maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
K = size(Dictionary,2);

% normalize dictionary
% do not need to normalize dictionary (why??)
%dnorm      = sqrt(sum(Dictionary(1:bb^2, :).^2, 1));
%ind        = find(dnorm ~= 0);    % remove zeros
%Dictionary(:,ind) = Dictionary(:,ind)./repmat(dnorm(ind), size(Dictionary,1), 1);

% normalize inputImg
%inputImg = inputImg./255;

%while (prod(floor((size(inputImg)-bb)/slidingDis)+1)>maxBlocksToConsider)
%    slidingDis        = slidingDis+1;
%    output.slidingDis = slidingDis;
%end
[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
[blocks2,idx2]  = my_im2col(inputImg2,[bb,bb],slidingDis);
blocks          = zeros(size(blocksraw));
blocksPredict   = zeros(size(blocksraw));
blocksError     = zeros(size(blocksraw));
predictError    = zeros(1,size(blocksraw,2));

%  normalize patch
%norms = sqrt(sum(blocksraw.^2));
%ind   = find(norms ~= 0);
%blocksraw(:,ind)                  = blocksraw(:,ind)./repmat(norms(1,ind),[size(blocksraw,1),1]);
%blocksraw(find(isnan(blocksraw))) = 0;

% batch processing with 30000
alphas = zeros(K,size(blocksraw,2));
%param.maxNumPatchesToTrainOn = 0;
if isfield(param, 'maxNumberofIterations')
    param.itermax = param.maxNumberofIterations;
end

param.numThreads = 4;

% for SLEP parameters
opts.init = 2;
opts.tFlag=5;
opts.maxIter = 100;
opts.nFlag = 1; % with normalization
opts.rFlag = 1;
opts.mFlag = 0;
opts.lFlag = 0;
lambda = 0.1;

funVal = zeros(opts.maxIter,size(blocksraw,2));
dnorm      = repmat(sqrt(sum(Dictionary(1:bb^2,:).^2)),[size(Dictionary(1:bb^2,:),1) 1]);
if param.lasso == 1
    %tic
    %alphas = mexLasso(blocksraw, Dictionary(1:bb^2,:), param);
    %alphas = mexCD(blocksraw, Dictionary(1:bb^2,:)./dnorm, sparse(size(Dictionary,2), ...
    %    size(blocksraw,2)), param);
    %toc
    
    for i = 1:size(blocksraw,2)
        [alphas(:,i), funVal(:,i)] = LeastR(Dictionary(1:bb^2,:)./dnorm, blocksraw(:,i), ...
            lambda, opts);
    end
else
    
    for jj = 1:30000:size(blocksraw,2)
        jumpSize = min(jj+30000-1,size(blocksraw,2));
        if (reduceDC)
            vecOfMeans = mean(blocksraw(:,jj:jumpSize));
            blocks(:,jj:jumpSize) = blocksraw(:,jj:jumpSize) - repmat(vecOfMeans,size(blocksraw,1),1);
        else
            blocks(:,jj:jumpSize) = blocksraw(:,jj:jumpSize);
        end
        
        % Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
        % Coefs = OMPerr(Dictionary(1:bb^2,:),blocks(:,jj:jumpSize),errT);
        param.maxNumPatchesToTrainOn = jumpSize-param.maxNumPatchesToTrainOn;
        
        % compute coefficient using sparse coding
        Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary(1:bb^2,:), param);
        alphas(:, jj:jumpSize) = Coefs;
        
        if (reduceDC)
            blocks(:,jj:jumpSize) = Dictionary(1:bb^2,:)*Coefs + ones(size(blocksraw,1),1) * vecOfMeans;
        else
            % blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs;
            blocks(:,jj:jumpSize) = Dictionary(1:bb^2,:)*Coefs;
        end
        
        if (reduceDC)
            vecMeans = param.vecMeans;
            rawPatch = param.rawPatch;
            randPermutation = randperm(size(rawPatch, 2));
            selectedPatches = randPermutation(1:10000);
            selectedRawPatch= rawPatch(:,selectedPatches);
            selectedVecMeans= vecMeans(:,selectedPatches);
            n2 = zeros(size(blocks,2), size(selectedRawPatch, 2));
            for j = 1:500:size(selectedRawPatch,2);
                %beginj = j;
                if j + 500 <= size(selectedRawPatch,2)
                    endj = j+500;
                else
                    endj = size(selectedRawPatch,2);
                end
                n2(:, j:endj) = dist2(blocks(:,jj:jumpSize)', selectedRawPatch(:,j:endj)');
            end
            [~, ind] = sort(n2,2);
            blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs + ones(size(blocksraw,1),1) * ...
                selectedVecMeans(2, ind(:,1)');
        else
            %blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs;
            blocksPredict(:,jj:jumpSize) = Dictionary(1+bb^2:end,:)*Coefs;
        end
    end
    
end

blocksPredict = Dictionary(1+bb^2:end,:)*alphas;
blocks        = Dictionary(1:bb^2,:)*alphas;

imin          = min(blocksPredict(:));
imax          = max(blocksPredict(:));
blocksPredict = (blocksPredict - imin)./(imax - imin);
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
