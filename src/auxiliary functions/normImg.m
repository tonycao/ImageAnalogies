function normedimg = normImg(img, border)

imgc = double(img(1+border:end-border, 1+border:end-border));

imax  = max(imgc(:));
imin  = min(imgc(:));
normedimg = (double(img)-imin)./(imax-imin);

%normedimgc = img;
%normedimgc(border:end-border, border:end-border) = normedimg(border:end-border, border:end-border);

normedimg(find(normedimg>1)) = 1;
normedimg(find(normedimg<0)) = 0;

%snormedimg = normedimgc;