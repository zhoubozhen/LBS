function value = FOCUS(image,type) 
% =========================================================================
% Focusing function.
% -------------------------------------------------------------------------
% Input:    -  image : The image to be computed.
%           -  type  : The required algorithm type.
% Output:   -  value : The value of the result.
% =========================================================================
[index_col,index_row]=find(image~=0);
zuoshang_row = min(index_row);
zuoshang_col = min(index_col);
row_length = max(index_row)-min(index_row);
col_length = max(index_col)-min(index_col);
rect = [zuoshang_row zuoshang_col row_length col_length];
image = imcrop(image,rect);
image = double(image);
switch type
    case "TC"
        jun = mean(image,"all");
        biao = std(double(image),0,"all");
        value = biao/jun;
    case "VAR1"
        I_mean = mean(image(:));
        I_new = (image-I_mean).^2;
        value = sum(I_new(:));
    case "LAP"
        sum_lap = 0;
        m = size(image,1);
        n = size(image,2);
        for i=2:m-1
            for j=2:n-1
                lap = image(i+1,j)+image(i-1,j)+image(i,j+1)+...
                image(i,j-1)-4*image(i,j);
                lap = lap^2;
                sum_lap = sum_lap+lap;
            end
        end
        value = sum_lap;
    case "diejia"
        value = sum(image(:));
end