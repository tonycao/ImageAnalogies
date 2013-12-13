close all
ind = [2 5 8 9 10 11];
trainingSize = length(ind);

for i = 1:trainingSize
    %sigma = 25;
    pathForImages ='data/';
    imgFileName1 = strcat('TEM', num2str(ind(i)), '01_edit_0.5.png');
    [IMin,pp]=imread(strcat([pathForImages,imgFileName1]));
    IMin = IMin(3:end-2,3:end-2);

    imgFileName2 = strcat('CONF', num2str(ind(i)), '01_green_registered_0.5.png');
    [IMin1,pp]=imread(strcat([pathForImages,imgFileName2]));
    IMin1 = IMin1(3:end-2,3:end-2);

    figure,
    subplot(1,2,1)
    imshow(IMin,[]);
    imgFileName1 = strcat('TEM', num2str(ind(i)), '01_edit_0.5_crop.png');
    imwrite(IMin,strcat([pathForImages,imgFileName1]));
    subplot(1,2,2)
    imshow(IMin1,[]);
    imgFileName2 = strcat('CONF', num2str(ind(i)), '01_green_registered_0.5_crop.png');
    imwrite(IMin1,strcat([pathForImages,imgFileName2]));
    
    pause
end
