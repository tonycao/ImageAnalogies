%%

%% dictionary learning test

% 1st dictionary learning test

setup
clear all
close all

profile on
% parameter initialization
pathForImages = 'data/TEMCONF/';
trainingSet   = {5,2};
param.maxNumberofDictionaryUpdate = 4;
param.maxNumBlocksToTrainOn       = 65000;
param.patchSideLength             = 10;
param.lambda                      = 0.1;
param.RR                          = 5;

i = 1;
count = 1;
for j = 1:6 % image index from 1 to 3
    if j~=i
        trainingSet{count,1}    = [pathForImages 'TEM' num2str(j) '.png'];
        trainingSet{count,2}    = [pathForImages 'CONF' num2str(j) '.png'];
        count = count + 1;
    end
end

outputDict = ['Dictionary/dl_new_' num2str(param.RR) '00_' num2str(i) '.mat'];

% disp(['learning dict for image pair ' num2str(i)]);
Dictionary = dictlearningTest(trainingSet, outputDict, param);


% display result
patchSize = param.patchSideLength^2;
dictSize = patchSize*param.RR;
figure, 
subplot(1,2,1)
displayDictionaryElementsAsImage(Dictionary(1:patchSize,:), uint8(sqrt(dictSize)), ...
    uint8(sqrt(dictSize)), param.patchSideLength, param.patchSideLength, 0);
subplot(1,2,2)
displayDictionaryElementsAsImage(Dictionary(1+patchSize:end,:), uint8(sqrt(dictSize)), ...
    uint8(sqrt(dictSize)), param.patchSideLength, param.patchSideLength, 0);


profile viewer