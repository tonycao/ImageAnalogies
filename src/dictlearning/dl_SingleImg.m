
function dl_SingleImg(imageind, runs)

% dictionary learning on single image with multple times
%
% Author: Tian Cao

setup
close all

pathForImages = 'data/TEMCONF/';
trainingSet = {1,2};
%param.maxNumberofDictionaryUpdate = 4;
param.lambda                      = 0.15;
param.patchSideLength  = 10; 
param.RR       = 10;
param.rotation = 12;
param.imageind = imageind;
count = 1;
for i = 1:runs
        
    param.runs = i;
    trainingSet{count,1}    = [pathForImages 'TEM' num2str(imageind) '.png'];
    trainingSet{count,2}    = [pathForImages 'CONF' num2str(imageind) '.png'];
    %count = count + 1;

    outputDict = ['Dictionary/dl_SingleImg_r' num2str(param.rotation) '_' num2str(param.patchSideLength) 'x' num2str(param.RR) ...
                  '_' num2str(imageind) '_' num2str(i) 's.mat'];
        
    disp(['learning dict for image pair ' num2str(imageind) 'the ' num2str(i) 'th time...'] );
    dictlearningTest(trainingSet, outputDict, param);
    
end