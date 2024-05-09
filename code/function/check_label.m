function [new_label] = check_label(this_label,last_label)
% =========================================================================
% Check if there are any new particles appearing in new reconstruction images.
% -------------------------------------------------------------------------
% Input:    - this_label  : The current incoming label image, namely the image to be processed.
%           - last_label  ：The checked label before..
% Output:   - new_label   : The checked label.
% =========================================================================
new_label = zeros(size(this_label));
num_particle = max(last_label(:));
for i=1:max(this_label(:))
    [this_row,this_col]= find(this_label==i);
    num_zuobiao = size(this_col);
    A = unique(sort(last_label(:)));
    if max(A(:))==0
        new_label = this_label;
        break;
    end
    for j=A(2):num_particle
        [last_row,last_col]=find(last_label==j);
        consider_matrix = ismember([this_row,this_col],[last_row,last_col],"rows");
        if max(consider_matrix(:)) % Whether it is all zero, all zero means that it is not found; If the return value is 1, it is found
            for k = 1:num_zuobiao
                new_label(this_row(k),this_col(k)) = j;
            end
            ok_or_not = 1; % found
            break;
        else
            ok_or_not = 0;
        end
    end
    if ok_or_not == 0 % not found
        num_particle = num_particle + 1; %Total number of particles found so far plus one
        for k = 1:num_zuobiao
            new_label(this_row(k),this_col(k)) = num_particle;
        end
    end
end