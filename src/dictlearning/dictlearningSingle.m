function [Dictionary] = dictlearningSingle(trainingSet, outputDict, param)

% test dictlearning
%
% Author: Tian Cao

%clear all
%close all
%clc

%setup
%profile on

param.save = 0;

if ~exist('param', 'var')
    param = struct;
end

% parameter setting

if ~isfield(param, 'rotation')
	param.rotation = 1; % rotate data to generate more training samples
end
	
if ~isfield(param, 'patchSideLength')
    param.patchSideLength = 10; % block size
end

if ~isfield(param, 'RR')
    param.RR              = 10; % redundancy factor
end
    
if ~isfield(param, 'sigma')
    param.sigma           = 1;
end

if ~isfield(param, 'lambda')
    param.lambda          = 0.5;
end

if ~isfield(param, 'rawpatchDict')
    param.rawpatchDict    = 0;
end

if ~isfield(param, 'mu')
    param.mu              = 1;
end

if ~isfield(param, 'numberofImages')
    param.numberofImages  = 2;
end
if ~isfield(param, 'slidingDis')
    param.slidingDis      = 1;
end

if ~isfield(param, 'lasso')
    param.lasso = 1;
end

if ~isfield(param, 'reducemean')
    param.reduceMean      = 0;
end

if ~isfield(param, 'dictSize')
    param.dictSize = param.patchSideLength^2*param.RR;
end

if ~isfield(param, 'mask')
	param.mask = 0;
end

if ~isfield(param, 'prediction')
    param.prediction = 0;
end

if ~isfield(param, 'verbose')
    param.verbose = 0;
end

trainingSize = size(trainingSet, 2);

param.trainingSize    = trainingSize*param.rotation;

Itraining = {};
if param.mask
	ItrainingMask = {};
end

for i = 1:trainingSize
    %read training image
    %imgFileName1 = strcat('TEM', num2str(ind(i)), '01_edit_0.5_crop.png');
    IMin    = trainingSet{2*(i-1)+1};
    IMin    = double(IMin(:,:,1));
    Imask   = zeros(size(IMin));
    
    % normalize image to (0,1)
    minval = min(IMin(:));
    maxval = max(IMin(:));
    IMin   = (IMin-minval)./(maxval-minval);
    if param.prediction
        IMin2    = trainingSet{2*(i-1)+2};
        IMin2    = double(IMin2(:,:,1));
        %Imask   = zeros(size(IMin2));
    
        % normalize image to (0,1)
        minval = min(IMin2(:));
        maxval = max(IMin2(:));
        IMin2   = (IMin2-minval)./(maxval-minval);
    end
    
    % 
    for j = 1:param.rotation
    	tmatrix = [cos(((j-1)/param.rotation)*2*pi)  sin(((j-1)/param.rotation)*2*pi) 0; ...
    	        -1*sin(((j-1)/param.rotation)*2*pi)  cos(((j-1)/param.rotation)*2*pi) 0; 0 0 1];
    	tform   = maketform('affine', tmatrix);
    	
    	IMinRot = imtransform(IMin, tform, 'bilinear', 'FillValues', 0);
        if param.prediction
            IMinRot2 = imtransform(IMin2, tform, 'bilinear', 'FillValues', 0);
        else
            IMinRot2 = IMinRot;
        end
        
        %IMinRot1= imtransform(IMin1, tform, 'bilinear', 'FillValues', 0);
        if param.mask
        	ImaskRot = imtransform(Imask, tform, 'bilinear', 'FillValues', 1);
        end
        
    	Itraining{((i-1)*param.rotation+j-1)*2+1} = IMinRot;
    	Itraining{((i-1)*param.rotation+j-1)*2+2} = IMinRot2;
    	if param.mask
            if isfield(param, 'immask')
                ItrainingMask{((i-1)*param.rotation+j-1)+1} = param.immask;
            else
                ItrainingMask{((i-1)*param.rotation+j-1)+1} = ImaskRot;
            end
    	end
    end
end


% parallel processing
tic
if param.mask
	[Dictionary] = dictionaryLearning_onlineMask(Itraining, ItrainingMask, param);
else
	[Dictionary] = dictionaryLearning_online(Itraining, param);
end
toc
%Dictionary = Dictionary(1:param.patchSideLength^2,:);

display = 0;
if display 
    ImD=displayPatches(Dictionary(1:param.patchSideLength^2,:));
    %figure,
    imagesc(ImD); colormap('gray');
end

if param.save == 1
save(outputDict, 'Dictionary', 'param');
disp('dictionary saved...');
end

%profile viewer
