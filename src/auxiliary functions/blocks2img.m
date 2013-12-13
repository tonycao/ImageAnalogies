% blocks to image

function img = blocks2img(blocks, idx, imgSize, param)

bb = param.patchSideLength;
slidingDis          = param.slidingDis;
[rows,cols]  = ind2sub(imgSize-bb+1,idx);
IMout  = zeros(imgSize);
Weight = zeros(imgSize);
% reconstruct image
count  = 1;
for i  = 1:length(cols)
    col   = cols(i);
    row   = rows(i);
    block = reshape(blocks(:,count),[bb,bb]);
    
    IMout(row:row+bb-1,col:col+bb-1)        = IMout(row:row+bb-1,col:col+bb-1)+block;
    Weight(row:row+bb-1,col:col+bb-1)       = Weight(row:row+bb-1,col:col+bb-1) + 1;%ones(bb);
    count = count+1;
end;

img = IMout./ Weight;