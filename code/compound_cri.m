% Introduction
% =========================================================================
%{
This program is for the compound focusing criterion used for the generation 
of reference result in the experiment.Run this program so you can get the
output reference result in the experiment. The result is named "output".
%}
% =========================================================================
% Author: Bozhen Zhou (zbz22@mails.tsinghua.edu.cn)
% =========================================================================
% Tips
% =========================================================================
%{
Change the string in line 27, "a1b1dis" for example, to get different
output reference result corresponding to different sub-holograms.

Make sure the source images' path is correct in line 28.
%}
clc;clear ; close all;
addpath .\function\
%% PSL algorithm to get the result dis1
label_bag = zeros(256,256,121);
add_image = zeros(256,256);
f = zeros(256,256,121);    % The pack for restoring all the reconstructed images
for dis = 1800:25:4800
    num = (dis-1800)/25+1; % The number of this reconstructed image
    str = "a1b1dis" + num2str(dis);
    str = "..\data\experiment data\recon_images_25μm\"+str+".bmp";
    test = imread(str);
    test = 255-test;
    f(:,:,num) = test;
    [result,label] = recognition(test); % call the function "recognition" to get the label marked by several different integers
    if num==1
        label_bag(:,:,num) = label;
        add_image = label;
    else
        this_label = label;
        last_label = add_image;
        if max(this_label(:))==0 && max(last_label(:))==0
            continue;
        end
        [new_label] = check_label(this_label,last_label); % call the function check_label，to get a new label
        label_bag(:,:,num) = new_label;                   % the check is over, then put them into the pack
        all_tip = unique(sort(new_label(:)));
        if max(all_tip)~=0
            for i= all_tip(2):max(all_tip)
                add_image(new_label==i) = i;
            end
        end
    end
end

% Calculate initial output result according to the label_bag
final_result = cell(max(label_bag(:)),1); % the radius, pixel radius, pixel size 2.2μm
final_depth = cell(max(label_bag(:)),1);  % depth in the z-direction
pos = cell(max(label_bag(:)),1);
lateral_pos = cell(max(label_bag(:)),1);  % lateral position

for i=1:121
    dis = 1800+(i-1)*25;
    the_label = label_bag(:,:,i);
    tips = unique(sort(the_label(:)));    % all label values
    if max(tips)~=0
        for j = tips(2):max(tips)
            that_label = zeros(256,256);  % initialization
            that_label(the_label==j)=1;
            stats = regionprops(that_label,"Area","Centroid","MajorAxisLength","MinorAxisLength");
            if isempty(stats)==0 
                position = stats(1).Centroid;
                diameters = mean([stats(1).MajorAxisLength stats(1).MinorAxisLength],2);
                radii = diameters*2.2;    
                final_result{j}=cat(2,final_result{j},radii);
                final_depth{j}=cat(2,final_depth{j},dis);
                lateral_pos{j} = cat(1,lateral_pos{j},position);
            end
        end
    end
end

inter_result = final_result;  
inter_lateral = lateral_pos;
for i =1:max(label_bag(:))
    pos{i} = find(final_result{i}==min(final_result{i}));
    final_depth{i} = final_depth{i}(round(median(pos{i})));
    final_result{i}= min(final_result{i}); 
    lateral_pos{i} = round(inter_lateral{i}(round(median(pos{i})),:));
end
figure;imshow(add_image,[]),title("the original distribution")

% Delete those that do not meet the criteria
deleted_add_image = add_image;
deleted_data = [];
for i= 1:max(add_image(:))
    this_pos = lateral_pos{i}; 
    number = size(inter_lateral{i},1); % number of discoveries
    if this_pos(1)<9 || this_pos(1)>247 || this_pos(2)<9 || ...
            this_pos(2)>247 || number<8
        deleted_add_image(add_image==i) = 0;
        deleted_data = cat(1,deleted_data,i);
    end
end
deleted_num = length(deleted_data); 
new_final_depth = cell(max(label_bag(:)-deleted_num),1);
new_lateral_pos = cell(max(label_bag(:)-deleted_num),1);
new_diameter = cell(max(label_bag(:)-deleted_num),1);
new_num = 0;
new_add_image = zeros(256,256);
for i =1:max(add_image(:))
    if ismember(i,deleted_data)==0
        new_num = new_num + 1;
        new_final_depth{new_num} = final_depth{i};
        new_lateral_pos{new_num} = lateral_pos{i};
        new_diameter{new_num} = final_result{i};
        new_add_image(deleted_add_image==i) = new_num;
    end
end
for i = 1:length(new_diameter)
    new_diameter{i} = round(new_diameter{i},1)*4; % keep one decimal place
end
figure;imshow(new_add_image,[]),title("new distribution")
[Nx,Ny,Nz] = size(f); 
o=zeros(Nx,Ny,Nz);
a=zeros(Nx,Ny);
for i=1:Nx
    for j=1:Ny
        a(i,j)=min(abs(f(i,j,:)));        % minimum intensity projection
        for k=1:Nz
            if  abs(f(i,j,k))<=a(i,j)*0.1 % filter noise
                f(i,j,k)=0;
            end
        end
    end
end
figure;imshow(a,[]),title("minimum intensity projection")
output.lateral_position = new_lateral_pos;
output.depth = new_final_depth;
output.diameter = new_diameter;

%% Apply TC algorithm to get the result dis2 and compare the two results to decide the final result
for i=1:length(new_final_depth)
    img = zeros(256,256);
    pos = output.lateral_position{i};
    if pos(1)<21
        xmin = 1;
        xmax = pos(1)+20;
    elseif pos(1)>235
        xmax = 256;
        xmin = pos(1)-20;
    else
        xmin = pos(1)-20;
        xmax = pos(1)+20;
    end
    if pos(2)<21
        ymin = 1;
        ymax = pos(2)+20;
    elseif pos(2)>235
        ymax = 256;
        ymin = pos(2)-20;
    else
        ymin = pos(2)-20;
        ymax = pos(2)+20;
    end
    img(ymin:ymax,xmin:xmax)=1;
    for j =1:Nz
        cover = img.*f(:,:,j);
        %imshow(cover,[])
        cal(j) =FOCUS(cover,"TC"); % TC algorithm
    end
    index = find(cal == max(cal(:)));
    new_dis = index*25+1775;
    old_dis = output.depth{i};
    difference = abs(old_dis-new_dis);
    if difference<150 && difference>50
        output.depth{i} = new_dis;
    elseif difference>250
        new_ind = (old_dis-1800)/25 + 1;
        for k = 1:10
            if new_ind+k-5 <121 && new_ind+k-5 >0
                cal2(k)= cal(new_ind+k-5);
            end
        end
        new_index = find(cal2 == max(cal2(:)))+new_ind-5;
        new_dis = new_index*25+1775;
        output.depth{i} = new_dis;
    end
end