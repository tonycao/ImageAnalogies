function [IOut, IOutPredict, output] = imageAnalogySCMask(inputImg, inputImg2, inputMask, Dictionary, param)

% image Analogy with Spare Representation model
%
%

%reduceDC            = param.reduceMean;
[NN1, NN2]          = size(inputImg);
bb                  = param.patchSideLength;
slidingDis          = param.slidingDis;
K = size(Dictionary,2);
lsregression = param.lsregression;

% normalize dictionary
dnorm      = sqrt(sum(Dictionary(1:bb^2, :).^2, 1));
ind        = find(dnorm ~= 0);    % remove zeros
Dictionary(:,ind) = Dictionary(:,ind)./repmat(dnorm(ind), size(Dictionary,1), 1);

% normalize inputImg
[blocksraw,idx] = my_im2col(inputImg,[bb,bb],slidingDis);
[blocks2,~]  = my_im2col(inputImg2,[bb,bb],slidingDis);
[blocksmask,~] = my_im2col(inputMask,[bb,bb],slidingDis);
%blocks          = zeros(size(blocksraw));
%blocksPredict   = zeros(size(blocksraw));
blocksError     = zeros(size(blocksraw));
%predictError    = zeros(1,size(blocksraw,2));

%alphas = zeros(K,size(blocksraw,2));
%sparsity = zeros(1,size(blocksraw,2));
param.numThreads = 4;
param.mode = 2;
%param.lambda2    = 0.1;
if param.prediction
    if ~isfield(param, 'L')
        param.L = 10;% maximum number of elements of each decomposition
    end
end
%% lasso with mask
%alphas = mexOMPMask(blocksraw(1:bb^2,:),Dictionary(1:bb^2,:),logical(abs(1-blocksmask)),param);
alphas = mexLassoMask(blocksraw(1:bb^2,:),Dictionary(1:bb^2,:),logical(abs(1-blocksmask)),param);
sparsity = sum(abs(alphas)>1e-6,1)./K;

% if lsregression
%     for i = 1:size(blocksraw,2)
%         ind = find(abs(alphas(:,i)) > 1e-6);
%         if ~isempty(ind)
%             dictTemp = bsxfun(@times, abs(1-blocksmask(:,i)), Dictionary(1:bb^2,:));
%             seldict = dictTemp(:,ind);
%             alphas(ind,i) = (seldict'*blocksraw(1:bb^2,i))\(seldict'*seldict);
%         end
%     end
% end

blocks = Dictionary(1:bb^2,:)*alphas.*abs(1-blocksmask);
blocksPredict = Dictionary(1+bb^2:end,:)*alphas.*abs(1-blocksmask);

% for i = 1:size(blocksraw,2)
% 	dictTemp = bsxfun(@times, abs(1-blocksmask(:,i)), Dictionary(1:bb^2,:));
% 	dictTemp2 = bsxfun(@times, abs(1-blocksmask(:,i)), Dictionary(1+bb^2:end,:));
% 	alphas(:,i) = mexLasso(blocksraw(1:bb^2,i), dictTemp, param);
%     
%     sparsity(i) = sum(abs(alphas(:,i))>1e-6)/K;
%     % least squares regrssion based on selected dict items
%     ind = [];
%     if lsregression
%         ind = find(abs(alphas(:,i)) > 1e-6);
%         if ~isempty(ind)
%             seldict = dictTemp(:,ind);
%             alphas(ind,i) = (seldict'*blocksraw(1:bb^2,i))\(seldict'*seldict);
%         end
%     end
%     
% 	blocks(:,i) = dictTemp*alphas(:,i);
% 	blocksPredict(:,i) = dictTemp2*alphas(:,i);
% end

totalpatch = sum(sum(abs(1-blocksmask),1)>0);
overallSparsity = sum(sparsity)/totalpatch;
if param.verbose
    disp(['sparsity level ' num2str(overallSparsity)]);
end

rectError     = sum((blocks - blocksraw).^2,1);
predictError  = sum((blocksPredict-blocks2).^2,1);
blocksError   = repmat(predictError,[size(blocksError,1) 1]);

output.alphas       = alphas;
output.predictError = predictError;
output.blocksError  = blocksError;
output.rectError    = rectError;
output.blocksmask = blocksmask;

count  = 1;
Weight = zeros(NN1,NN2);
IMout  = zeros(NN1,NN2);
IMoutPredict = zeros(size(IMout));
%IMoutError   = zeros(size(IMout));
[rows,cols]  = ind2sub(size(inputImg)-bb+1,idx);

% reconstruct image
for i  = 1:length(cols)
    col   = cols(i); row   = rows(i);
    block = reshape(blocks(:,count),[bb,bb]);
    blockPredict = reshape(blocksPredict(:,count),[bb,bb]);
    %blockError   = reshape(blocksError(:,count),[bb, bb]);
    
    IMout(row:row+bb-1,col:col+bb-1)        = IMout(row:row+bb-1,col:col+bb-1)+block;
    IMoutPredict(row:row+bb-1,col:col+bb-1) = IMoutPredict(row:row+bb-1,col:col+bb-1) + blockPredict;
    %IMoutError(row:row+bb-1,col:col+bb-1)  = IMoutError(row:row+bb-1,col:col+bb-1) + blockError;
    Weight(row:row+bb-1,col:col+bb-1)       = Weight(row:row+bb-1,col:col+bb-1) + 1;%ones(bb);
    %WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1) + 1;%ones(bb);
    count = count+1;
end;

%IOut  = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);
lambda = 0.02;
%IOut  = (lambda*inputImg+IMout)./(lambda*1+Weight);
IOut   = IMout./Weight;
IOutPredict = IMoutPredict./Weight;
%IoutError  = IMoutError ./ Weight;
%output.IoutError = IoutError;
%output.Weight    = Weight;
%figure, imshow(IoutError, []);
%save('IoutError.mat', 'IoutError');
