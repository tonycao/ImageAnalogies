function [IOut, IOutPredict, output] = imageAnalogySCwithMean(inputImg, inputImg2, Dictionary, param)

reduceDC = 1; %param.reduceMean;
[NN1, NN2] = size(inputImg);
bb = param.patchSideLength;
slidingDis = param.slidingDis;
maxBlocksToConsider = param.maxBlocksToConsider;
%maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
K = size(Dictionary,2);


while (prod(floor((size(inputImg)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
    output.slidingDis = slidingDis;
end
[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
[blocks2]     = my_im2col(inputImg2,[bb,bb],slidingDis);
blocks        = zeros(size(blocksraw));
blocksPredict = zeros(size(blocksraw)); 
blocksError   = zeros(size(blocksraw));
predictError  = zeros(1,size(blocksraw,2));
blockswithMean= zeros(size(blocksraw,1)+1, size(blocksraw,2));

% go with jumps of 30000
alphas = zeros(K,size(blocksraw,2));
param.maxNumPatchesToTrainOn = 0;
for jj = 1:30000:size(blocksraw,2)
    jumpSize = min(jj+30000-1,size(blocksraw,2));
    if (reduceDC)
        vecOfMeans = mean(blocksraw(:,jj:jumpSize));
        % decouple mean intensity and variance of image patch
        blocks(:,jj:jumpSize) = blocksraw(:,jj:jumpSize) - repmat(vecOfMeans,size(blocksraw,1),1);
        blockswithMean(:,jj:jumpSize) = [blocks(:,jj:jumpSize);vecOfMeans];
    end
    
    % Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT);
    % Coefs = OMPerr(Dictionary(1:bb^2,:),blocks(:,jj:jumpSize),errT);
    param.maxNumPatchesToTrainOn = jumpSize-param.maxNumPatchesToTrainOn;
    Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary(1:bb^2,:), param);
    alphas(:, jj:jumpSize) = Coefs;
    if (reduceDC)
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs + ones(size(blocksraw,1),1) * vecOfMeans;
    else
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs ;    
    end
    if (reduceDC)
        %{
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
        [ns, ind] = sort(n2,2);
        blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs + ones(size(blocksraw,1),1) * ...
            selectedVecMeans(2, ind(:,1)');
        %}
        blocksPredict(:,jj:jumpSize)= Dictionary(2+bb^2:end-1,:)*Coefs + ones(size(blocksPredict,1),1)*Dictionary(end,:)*Coefs;
    else 
        blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs;
    end
end

imin = min(blocksPredict(:));
imax = max(blocksPredict(:));
blocksPredict = (blocksPredict - imin)./(imax - imin);
predictError = sum((blocksPredict-blocks2).^2,1);
threshold = 28;
blocksError = repmat(predictError,[size(blocksError,1) 1]);

%        predictError(jj:jumpSize) = sum((blocksPredict(:,jj:jumpSize)-blocks2(:,jj:jumpSize)).^2,1); 
%        blocksError(:,jj:jumpSize) = repmat(predictError(jj:jumpSize),[size(blocksError,1) 1]);

output.alphas = alphas;
output.predictError = predictError;
output.blocksError = blocksError;

count = 1;
Weight = zeros(NN1,NN2);
%WeightPredict = zeros(size(Weight));
IMout = zeros(NN1,NN2);
IMoutPredict = zeros(size(IMout));
IMoutError = zeros(size(IMout));
[rows,cols] = ind2sub(size(inputImg)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);        
    block =reshape(blocks(:,count),[bb,bb]);
    blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    blockError = reshape(blocksError(:,count),[bb, bb]);
    
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    IMoutPredict(row:row+bb-1,col:col+bb-1)=IMoutPredict(row:row+bb-1,col:col+bb-1)+blockPredict;
    IMoutError(row:row+bb-1,col:col+bb-1) = IMoutError(row:row+bb-1,col:col+bb-1) + blockError;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    %WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

%IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
lambda = 1;
IOut = (lambda*inputImg+IMout)./(lambda*1+Weight);
IOutPredict = IMoutPredict./Weight;
IoutError = IMoutError ./ Weight;
output.IoutError = IoutError;
output.Weight = Weight;
%figure, imshow(IoutError, []);
%save('IoutError.mat', 'IoutError');
