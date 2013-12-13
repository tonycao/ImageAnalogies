% test image analogy using sparse model

clear all
close all
%clc

setup
profile on

CombineDict = [];
load('Dictionary/dl_new_500_1.mat');
%load('fiducial_dictionary15_1000.mat');
%load('Fiducial_dictionary10x10_dn_raw.mat');
CombineDict  = [Dictionary CombineDict];


lambda = 0.1;
%error = zeros(6, length(lambdas));
for i = 2:6%[2 5 8 9 10 11];%]
        % tem/conf
        pathForImages = 'data/TEMCONF/';
        imageName = ['TEM' num2str(i) '_0.5.png'];
        imageName2 = ['CONF' num2str(i) '_0.5.png'];
        
        % sem/conf
        %pathForImages = 'data/Fiducial/';
        %imageName = ['source' num2str(i) '_DNSP_0.2.png'];
        %imageName2 = ['target' num2str(i) '_DNSP_0.2.png'];
        
        %read and normalize input
        inputImg = double(imread(strcat([pathForImages, imageName])));
        maxval = max(inputImg(:));
        minval = min(inputImg(:));
        inputImg = (inputImg-minval)./(maxval-minval);
        
        %read and normalize input
        inputImg2 = double(imread(strcat([pathForImages, imageName2])));
        maxval = max(inputImg2(:));
        minval = min(inputImg2(:));
        inputImg2N = (inputImg2-minval)./(maxval-minval);
        
        param.numberofImages = 1;
        param.gamma = 1;
        param.lambda = lambda;
        param.sigma = 1;
        param.maxNumberofIterations = 1000;
        psl = param.patchSideLength;
        param.reduceMean = 0;
        param.lasso = 1;
        param.slidingDis = 1;
       
        
        displayDic = 0;
        if displayDic == 1
            K = param.dictSize;
            bb = param.patchSideLength;
            figure, subplot(1,2,1);
            I = displayDictionaryElementsAsImage(CombineDict(1:bb^2,:), floor(sqrt(K)), ...
                floor(size(CombineDict,2)/floor(sqrt(K))),bb,bb,0);
            subplot(1,2,2);
            I = displayDictionaryElementsAsImage(CombineDict(1+bb^2:end,:), floor(sqrt(K)), ...
                floor(size(CombineDict,2)/floor(sqrt(K))),bb,bb,0);
        end
        
        tic
        %matlabpool
        [imgdenoise, imgPredict, output] = imageAnalogySC(inputImg, inputImg2N, CombineDict, param);
        toc
        
        %imwrite(imgPredict,'TEM501_edit_predict.png')
        imax  = max(imgPredict(:));
        imin  = min(imgPredict(:));
        I = uint8((imgPredict-imin)/(imax-imin)*255);
        N = hist(inputImg2(:), 0:255);
        K = histeq(I,N);
        
        figure,
        subplot(1,3,1);
        imshow(inputImg,[]);
        title('TEM image');
        subplot(1,3,2);
        imshow(uint8(inputImg2));
        title('Confocal image');
        subplot(1,3,3);
        imshow(K);
        title('Predicted confocal image');
        
end

%plot(error);
profile viewer