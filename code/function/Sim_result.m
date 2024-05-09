function final_pre_result = Sim_result(type,group_num,path,dis_start,dis_interval,dis_end,added_dia)
added_num = added_dia * 4;
switch type
    case 'pr'
        if added_num>=0
        final_pre_result = [];
        pre_num = 0;
        for dis = dis_start:dis_interval:dis_end 
            head = path+'\';
            pre_path =  sprintf("pr%d_dis",group_num); 
            pre_path = strcat( head , pre_path , num2str(dis) , ".bmp" );
            if ~exist (pre_path)
                break;
            end
            pre_img = imread(pre_path);
            fi = pre_img;
            fi(fi<50)=0;
            fi(fi>150+added_num)=0;
            bw_fi = imbinarize(fi);
            [B,L] = bwboundaries(bw_fi);
            stats = regionprops(L,"Centroid");
            result = [];
            if ~isempty(B)
                for i = 1:length(B)
                    X = round(stats(i).Centroid(1));
                    Y = round(stats(i).Centroid(2));
                    grey_value = round(mean([fi(Y-1,X-1),fi(Y-1,X),fi(Y-1,X+1), ...
                        fi(Y,X-1),fi(Y,X),fi(Y,X+1),fi(Y+1,X-1),fi(Y+1,X),fi(Y+1,X+1)]));
                    if grey_value<(125+added_num) && grey_value>(75)
                        pre_num = pre_num + 1 ;
                        result = cat(1,result,[X,Y,dis,grey_value]);
                    end
                end
                if all(max(result(:))~=0)
                    final_pre_result = cat(1,final_pre_result,result);
                end
            end
        end
        elseif added_num<0
            final_pre_result = [];
        pre_num = 0;
        for dis = dis_start:dis_interval:dis_end 
            head = path+'\';
            pre_path =  sprintf("pr%d_dis",group_num); 
            pre_path = strcat( head , pre_path , num2str(dis) , ".bmp" );
            if ~exist (pre_path)
                break;
            end
            pre_img = imread(pre_path);
            fi = pre_img;
            fi(fi<50+added_num)=0;
            fi(fi>150)=0;
            bw_fi = imbinarize(fi);
            [B,L] = bwboundaries(bw_fi);
            stats = regionprops(L,"Centroid");
            result = [];
            if ~isempty(B)
                for i = 1:length(B)
                    X = round(stats(i).Centroid(1));
                    Y = round(stats(i).Centroid(2));
                    grey_value = round(mean([fi(Y-1,X-1),fi(Y-1,X),fi(Y-1,X+1), ...
                        fi(Y,X-1),fi(Y,X),fi(Y,X+1),fi(Y+1,X-1),fi(Y+1,X),fi(Y+1,X+1)]));
                    if grey_value<(125) && grey_value>(75+added_num)
                        pre_num = pre_num + 1 ;
                        result = cat(1,result,[X,Y,dis,grey_value]);
                    end
                end
                if all(max(result(:))~=0)
                    final_pre_result = cat(1,final_pre_result,result);
                end
            end
        end
        end
    case 'truth'
        final_pre_result = [];
        pre_num = 0;
        for dis = dis_start:dis_interval:dis_end 
            head = path+'\';
            allfile = dir(head);
            example_str = allfile(3).name;
            num_pattern = '(\d+)Num';
            num_match = regexp(example_str, num_pattern, 'tokens', 'once');
            particle_num = uint8(str2double(num_match{1}));
            slice_pattern = '(\d+)Slice';
            slice_match = regexp(example_str, slice_pattern, 'tokens', 'once');
            slice_num = uint8(str2double(slice_match{1}));
            pre_path = sprintf('pr%d_%dNum_%dSlice_at%dum.bmp',group_num,particle_num,slice_num,dis);
            pre_path = strcat( head , pre_path );
            if ~exist (pre_path)
                break;
            end
            pre_img = imread(pre_path);
            fi = pre_img;
            fi(fi<50+added_num)=0;
            fi(fi>150+added_num)=0;
            bw_fi = imbinarize(fi);
            [B,L] = bwboundaries(bw_fi);
            stats = regionprops(L,"Centroid");
            result = [];
            if ~isempty(B)
                for i = 1:length(B)
                    X = round(stats(i).Centroid(1));
                    Y = round(stats(i).Centroid(2));
                    grey_value = round(mean([fi(Y-1,X-1),fi(Y-1,X),fi(Y-1,X+1), ...
                        fi(Y,X-1),fi(Y,X),fi(Y,X+1),fi(Y+1,X-1),fi(Y+1,X),fi(Y+1,X+1)]));
                    if grey_value<(125+added_num) && grey_value>(75+added_num)
                        pre_num = pre_num + 1 ;
                        result = cat(1,result,[X,Y,dis,grey_value]);
                    end
                end
                if all(max(result(:))~=0)
                    final_pre_result = cat(1,final_pre_result,result);
                end
            end
        end
end