function [Iout, output] = denoiseSparseCoding(image,dictionary,param)

reduceMean = 1;
waitBarOn = 1;
maxPatchesToConsider = param.maxPatchesToConsider;
patchLength = param.patchLength;
slidingDis = param.slidingDis;
[imSize1, imSize2] = size(image);

while (prod(floor((size(image)-patchLength)/slidingDis)+1)>maxPatchesToConsider)
    slidingDis = slidingDis+1;
end
[patches, idx] = my_im2col(image, [patchLength, patchLength], slidingDis);

if (waitBarOn)
    newCounterForWaitBar = (param.numIteration+1)*size(patches,2);
end

% 
for jj = 1:30000:size(blocks,2)
    if (waitBarOn)
        waitbar(((param.numIteration*size(patches,2))+jj)/newCounterForWaitBar);
    end
    jumpSize = min(jj+30000-1,size(patches,2));
    if (reduceMean)
        vecMeans = mean(patches(:,jj:jumpSize));
        patches(:,jj:jumpSize) = patches(:,jj:jumpSize)-repmat(vecMeans,size(patches,1),1);
    end
    
    alphas = sparseCoding(patches, dictionary);
    if (reduceMean)
        patches(:,jj:jumpSize) = dictionary*alphas + ones(size(patches,1),1)*vecMeans;
    else
        patches(:,jj:jumpSize) = dictionary*alphas;
    end
end

count = 1;
IMout = zeros(imSize1, imSize2);
weight = zeros(imSize1, imSize2);
[rows, cols] = ind2sub(size(image)-patchLength+1,idx);
for i = 1:length(cols)
    col = cols(i); row = rows(i);
    patch = reshape(patches(:,count),[patchLength,patchLength]);
    IMout(row:row+patchLength-1,col:col+patchLength-1)=IMout(row:row+patchLength-1,...
        col:col+patchLength-1)+patch;
    Weight(row:row+patchLength-1,col:col+patchLength-1)=Weight(row:row+patchLength-1,...
        col:col+patchLength-1)+ones(patchLength);
    count = count+1;
end

if (waitBarOn)
    close(h);
end
lambda = param.lambda;
Iout = (lambda.*Image+IMout)./(lambda+Weight);
