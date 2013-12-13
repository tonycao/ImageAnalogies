function [Iout, IOutPredict, output] = imageAnalogySC(inputImg, Dictionary, param)

reduceDC = 0;
[NN1, NN2] = size(inputImg);
bb = param.patchSideLength;
slidingDis = 1;
maxBlocksToConsider = param.maxBlocksToConsider;
maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
K = param.dictSize;

while (prod(floor((size(inputImg)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
blocksPredict = zeros(size(blocks)); 

% go with jumps of 30000
alphas = zeros(K,size(blocks,2));
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
    Coefs = sparsecoding(blocks(:,jj:jumpSize), Dictionary(1:bb^2,:), param);
    alphas(:, jj:jumpSize) = Coefs;
    if (reduceDC)
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs + ones(size(blocks,1),1) * vecOfMeans;
    else
        blocks(:,jj:jumpSize)= Dictionary(1:bb^2,:)*Coefs ;
        blocksPredict(:,jj:jumpSize)= Dictionary(1+bb^2:end,:)*Coefs;
    end
end

output.alphas = alphas;
count = 1;
Weight = zeros(NN1,NN2);
WeightPredict = zeros(size(Weight));
IMout = zeros(NN1,NN2);
IMoutPredict = zeros(size(IMout));
[rows,cols] = ind2sub(size(inputImg)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);        
    block =reshape(blocks(:,count),[bb,bb]);
    blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    IMoutPredict(row:row+bb-1,col:col+bb-1)=IMoutPredict(row:row+bb-1,col:col+bb-1)+blockPredict;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

%IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
IOut = (inputImg+IMout)./(1+Weight);
IOutPredict = IMoutPredict./WeightPredict;
