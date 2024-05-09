% Introduction
% =========================================================================
%{
This program can delete additional particles overlapping in the axial
direction.
%}
% =========================================================================
% Author: Bozhen Zhou (zbz22@mails.tsinghua.edu.cn)
% =========================================================================
% Tips
% =========================================================================
%{
Run the "compare_exp.mlapp" first to get the data.

The "sorted_truth" is the data after deletion.
Inside the "negative_pack" are the FP particles.
Inside the "omitted_pack" are the FN particles.
Inside the "positive_pack" are the TP particles.
%}
data = sortrows(pre_result_pack,[5,6,1,2,3]); 
num = size(data,1);
keep_row = true(size(data, 1), 1);
filtered_data = [];
x_lab = data(1,1);
y_lab = data(1,2);
group = data(1,:);
% Find the data in which column 5 and column 6 are the same, and the
% absolute difference between column 1 and column 2 is within 3
for i=1:num
    x_new = data(i,1);
    y_new = data(i,2);
    if abs(x_new-x_lab)<3 && abs(y_new-y_lab)<3
        group = cat(1,group,data(i,:));
        continue;
    else
        x_lab = x_new;
        y_lab = y_new;
        group_num = size(group,1);
        new_group = sortrows(group,3);
        row_with_median = new_group(round((group_num+1)/2),:);
        filtered_data = cat(1,filtered_data,row_with_median);
        group = data(i,:);
    end
end

sorted_truth = sortrows(truth_result_pack, [5, 6, 3]);
XY_EEROR = 10;
Z_EEROR = 151;
pre_num = size(filtered_data,1);
truth_num = size(sorted_truth,1);
positive_pack = [];
negative_pack = [];
omitted_pack = [];
for i=1:pre_num
    X = filtered_data(i,1);
    Y = filtered_data(i,2);
    Z = filtered_data(i,3);
    AA = filtered_data(i,5);
    BB = filtered_data(i,6);
    for j=1:truth_num
        X0 = sorted_truth(j,1);
        Y0 = sorted_truth(j,2);
        Z0 = sorted_truth(j,3);
        AA0 = sorted_truth(j,5);
        BB0 = sorted_truth(j,6);
        if (AA == AA0)&&(BB == BB0)&&(abs(X-X0)<XY_EEROR) && (abs(Y-Y0)<XY_EEROR) && (abs(Z-Z0)<Z_EEROR)
            positive_pack = cat(1,positive_pack,filtered_data(i,:));
            break;
        elseif j == truth_num
            negative_pack = cat(1,negative_pack,filtered_data(i,:));            
        end
    end
end

for i=1:truth_num
    X0 = sorted_truth(i,1);
    Y0 = sorted_truth(i,2);
    Z0 = sorted_truth(i,3);
    AA0 = sorted_truth(i,5);
    BB0 = sorted_truth(i,6);
    for j=1:pre_num
        X = filtered_data(j,1);
        Y = filtered_data(j,2);
        Z = filtered_data(j,3);
        AA = filtered_data(j,5);
        BB = filtered_data(j,6);
        if (AA == AA0)&&(BB == BB0)&&(abs(X-X0)<XY_EEROR) && (abs(Y-Y0)<XY_EEROR) && (abs(Z-Z0)<Z_EEROR)            
            break;
        elseif j == pre_num
            omitted_pack = cat(1,omitted_pack,sorted_truth(i,:));
        end
    end
end