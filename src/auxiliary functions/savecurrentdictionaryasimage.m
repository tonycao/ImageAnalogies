function savecurrentdictionaryasimage(outputFileNamePrefix, matDictionary, param)

nrofPatches(1) = ceil(sqrt(param.dictSize));
nrofPatches(2) = nrofPatches(1);

dicPatchImageSize(1) = nrofPatches(1)*param.patchSideLength;
dicPatchImageSize(2) = dicPatchImageSize(1);

% initialize patch images
dictionaryPatchImages = zeros(dicPatchImageSize(1), dicPatchImageSize(2), ...
                                param.numberofImages);
for iImg = 1:param.numberofImages   
for iDic = 1:param.dictSize
    
    % patch position
    iXP = mod((iDic-1),nrofPatches(1));
    iYP = floor((iDic-1)/nrofPatches(1));
    
    % patch image coordinates
    iXC = iXP*param.patchSideLength;
    iYC = iYP*param.patchSideLength;
    iXCOffset = iXC + param.patchSideLength;
    iYCOffset = iYC + param.patchSideLength;
    % copy patch to image
    
    dictionaryPatchImages((iXC+1):iXCOffset,(iYC+1):iYCOffset,iImg) = reshape(matDictionary(param.patchSideLength^2*(iImg-1)+1:param.patchSideLength^2*iImg,iDic), ...
                            param.patchSideLength, param.patchSideLength);
    %dictionaryPatchImages((iXC+1):iXCOffset,(iYC+1):iYCOffset,2) = reshape(matDictionary(param.patchSideLength^2+1:end,iDic), ...
    %                        param.patchSideLength, param.patchSideLength);
end
end
for iI = 1:param.numberofImages
    outputFileNameStream = sprintf('image-%03d-%s.png', iI, outputFileNamePrefix);
    writeimageautoscale(dictionaryPatchImages(:,:,iI), outputFileNameStream, ...
        0, 255, 0, 0);
    
    fprintf('min = %d, max = %d\n', min(min(dictionaryPatchImages(:,:,iI))),...
        max(max(dictionaryPatchImages(:,:,iI))));
end

