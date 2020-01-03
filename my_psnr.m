function res = my_psnr_mask(f,g,mask,dynamicRangeMax,dynamicRangeMin)
% res = my_psnr(f,g,dynamicRangeMax,dynamicRangeMin)
% f: ref img
% g: distorted img
if nargin<3
    mask = ones(size(f));
end

F = im2double(f); % original
G = im2double(g); % distorted
if nargin<=2
    dynamicRangeMax = max(g(mask>0));
    dynamicRangeMin = min(g(mask>0));
end
if nargin<=3
    dynamicRangeMin = min(g(mask>0));
end
F(F>dynamicRangeMax) = dynamicRangeMax;
G(G>dynamicRangeMax) = dynamicRangeMax;
F(F<dynamicRangeMin) = dynamicRangeMin;
G(G<dynamicRangeMin) = dynamicRangeMin;
N = dynamicRangeMax - dynamicRangeMin;

mse = mean((F(mask>0)-G(mask>0)).^2);
res = 10*log10( N^2 / mse ); % = 1 / (1/N*sum(E(:).^2)!
end