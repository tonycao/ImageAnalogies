function [Dictionary] = dictlearningTest(trainingSet, outputDict, param)

% test dictlearning
%
% Author: Tian Cao


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

if ~isfield(param, 'dicSize')
    param.dictSize        = param.RR*param.patchSideLength^2; % number of atoms in the dictionary
end
%param.maxNumPatchesToTrainOn = 5*param.patchSideLength^2;

if ~isfield(param, 'sigma')
    param.sigma           = 1;
end

if ~isfield(param, 'lambda')
    param.lambda          = 0.2;
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

if ~isfield(param, 'normalized')
    param.normalized      = false;
end

if ~isfield(param, 'reducemean')
    param.reduceMean      = 0;
end

if ~isfield(param, 'saveIntermediateDictionaryResults')
    param.saveIntermediateDictionaryResults = false;
end

if ~isfield(param, 'relativeEnergyStoppingTolerance')
    param.relativeEnergyStoppingTolerance   = 1e-4;
end

if ~isfield(param, 'absoluteEnergyStoppingTolerance')
    param.absoluteEnergyStoppingTolerance   = 1e-2;
end

if ~isfield(param, 'maxNumberofIterations')
    param.maxNumberofIterations             = 2000;
end

if ~isfield(param, 'maxNumberofDictionaryUpdate')
    param.maxNumberofDictionaryUpdate       = 10;
end

if ~isfield(param, 'maxNumBlocksToTrainOn')
    param.maxNumBlocksToTrainOn             = 65000;
end

if ~isfield(param, 'maxBlocksToConsider')
    param.maxBlocksToConsider               = 100000;
end

if ~isfield(param, 'lasso')
    param.lasso = 1;
end

if ~isfield(param, 'imageind')
    param.imageind = 1;
end

if ~isfield(param, 'runs')
    param.runs = 1;
end

%ind = [2 8 9 10 11];
trainingSize = size(trainingSet, 1);

%if ~isfield(param, 'trainingSize')
param.trainingSize    = trainingSize*param.rotation;
%end

Itraining = {};

for i = 1:trainingSize
    %sigma = 25;
    %pathForImages ='';
    
    %read training image
    %imgFileName1 = strcat('TEM', num2str(ind(i)), '01_edit_0.5_crop.png');
    [IMin,pp]    = imresize(imread(trainingSet{i,1}),0.5);
    IMin         = double(IMin(:,:,1));
    
    % normalize image to (0,1)
    minval = min(IMin(:));
    maxval = max(IMin(:));
    IMin   = (IMin-minval)./(maxval-minval);
    
    % read training image
    %imgFileName2 = strcat('CONF', num2str(ind(i)), '01_green_registered_0.5_crop.png');
    [IMin1,pp]   = imresize(imread(trainingSet{i,2}),0.5);
    IMin1        = double(IMin1(:,:,1));
    
    % normalize image to (0,1)
    minval = min(IMin1(:));
    maxval = max(IMin1(:));
    IMin1  = (IMin1-minval)./(maxval-minval);
    
    %imSize = size(IMin);
    
    %
    for j = 1:param.rotation
        tmatrix = [cos(((j-1)/param.rotation)*2*pi)  sin(((j-1)/param.rotation)*2*pi) 0; ...
            -1*sin(((j-1)/param.rotation)*2*pi)  cos(((j-1)/param.rotation)*2*pi) 0; 0 0 1];
        tform   = maketform('affine', tmatrix);
        
        IMinRot = imtransform(IMin, tform, 'bilinear', 'FillValues', 0);
        IMinRot1= imtransform(IMin1, tform, 'bilinear', 'FillValues', 0);
        
        Itraining{((i-1)*param.rotation+j-1)*2+1} = IMinRot;
        Itraining{((i-1)*param.rotation+j-1)*2+2} = IMinRot1;
        
        %outputDictTemp = ['Dictionary/dl_SingleImg_r' num2str(param.rotation) '_' num2str(param.patchSideLength) 'x' num2str(param.RR) ...
        %          '_' num2str(param.imageind) '_' num2str(param.runs) '_s' num2str(j) 'r.mat'];
        %ItrainingTemp{1} = IMinRot;
        %ItrainingTemp{2} = IMinRot1;
        %tic
        %[Dictionary] = dictionaryLearning_online(ItrainingTemp, param);
        %toc
        %save(outputDictTemp, 'Dictionary', 'param');
        %disp('temp dictionary saved...');
    end
end


% parallel processing
tic

%[Dictionary] = dictionaryLearning(Itraining, param);
[Dictionary] = dictionaryLearning_online(Itraining, param);
toc

save(outputDict, 'Dictionary', 'param');
disp('dictionary saved...');

%profile viewer
