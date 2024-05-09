function [hlo,xt,H] = Backpropagation(z1,obj,pix_pitch,lambda)
xt=obj;
[objx, objy]=size(obj);
um=1e-6;
mm=1e-3;
k=2*pi/lambda; %wave_number
%% the captured hologram's specification
obj_dime_x=objx*pix_pitch;
obj_dime_y=objy*pix_pitch;

%% sampling for convolution method
[f_x, f_y]=sampl_freq(objx, objy, obj_dime_x, obj_dime_y); % convolution sampling
%f_x(1,1)
%% convolution method
[hlo,H]=FrPr_conlu(obj, lambda, z1, f_x, f_y);
hlo_abs=abs(hlo);
hlo=conj(hlo);
% ["hlo", num2str(hlo(1,1))]
g=1+abs(hlo).^2+conj(hlo)+hlo; % captured hologram by camera.
g_temp=abs(g).^2;
g_temp=g_temp*255/max(max(g_temp));
end

%% Convolution method
function [f_x, f_y]=sampl_freq(X, Y, x_dimen, y_dimen)
[f_x,f_y]=meshgrid((-X/2)*(1/x_dimen):(1/x_dimen):(X/2-1)*(1/x_dimen),(-Y/2)*(1/y_dimen):(1/y_dimen):(Y/2-1)*(1/y_dimen));
t=0;
end

function [hlo, H]=FrPr_conlu(obj, lambda, d, f_x, f_y)
k=2*pi/lambda;
H=exp(1i*k*d)*exp(-1i*pi*lambda*d*(f_x.^2+f_y.^2)); % without mask
H_temp=ifftshift(H);
O=fft2(ifftshift(obj));
hlo_temp=H_temp.*O;
hlo=fftshift(ifft2(hlo_temp));
t=abs(conj(hlo))-abs(conj(hlo));
end