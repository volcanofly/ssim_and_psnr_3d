function [mssim_3d] = ssim_3d(refimg, subimg,mask)
if nargin<3
    mask = ones(size(refimg));
end

img1 = double(refimg);
img2 = double(subimg);

% img2 = img2/mean(img2(mask>0))*mean(img1(mask>0));
% img1 = img1/max(img1(:))*255;
% img2 = img2/max(img1(:))*255;
% img2(img2>255)=255;

img1 = img1/max(img1(:))*255;
img2 = img2/mean(img2(mask>0))*mean(img1(mask>0));
img2(img2>255)=255;

[M,N,P]=size(img1);
% automatic downsampling
f = max(1,round(min([M,N,P])/256));
%downsampling by f
%use a simple low-pass filter 
if(f>1)
    lpf = ones(f,f,f);
    lpf = lpf/sum(lpf(:));
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');

    img1 = img1(1:f:end,1:f:end,1:f:end);
    img2 = img2(1:f:end,1:f:end,1:f:end);
end

[~,ssim_map_3d]=ssim(img2,img1);
mssim_3d=mean(ssim_map_3d(mask>0));
return