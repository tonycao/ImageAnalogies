
function writeimageautoscale(Image, outputFileNameStream, ...
    outputMin, outputMax, scale, y)
        
inputMin = min(Image(:));
inputMax = max(Image(:));

% compute scaling factors

scale = (outputMax - outputMin)/(inputMax - inputMin);
y = -inputMin*scale + outputMin;

% scale image
scaledDictionaryPatchImage = Image.*scale+y;
indlMin = find(scaledDictionaryPatchImage<outputMin);
indbMax = find(scaledDictionaryPatchImage>outputMax);
scaledDictionaryPatchImage(indlMin) = repmat(outputMin, size(indlMin), 1);
scaledDictionaryPatchImage(indbMax) = repmat(outputMax, size(indbMax), 1);

imwrite(uint8(scaledDictionaryPatchImage), outputFileNameStream);

