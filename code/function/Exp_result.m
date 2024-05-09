function final_pre_result = Exp_result(aa,bb,path,dis_start,dis_interval,dis_end)
% =========================================================================
% The output processing algorithm: convert the image information to the particle field information.
% -------------------------------------------------------------------------
% Input:    -  aa,bb                            : The serial number in the lateral direction.
%           -  dis_start, dis_interval, dis_end : The distance in μm. 
% Output:   -  final_pre_result                 : The particle field information.
% =========================================================================
final_pre_result = [];
pre_num = 0;
for dis = dis_start:dis_interval:dis_end 
    head = path;
    pre_path =  sprintf("pra%db%d_dis",aa,bb); 
    pre_path = strcat( head , pre_path , num2str(dis) , ".bmp" );
    if ~exist (pre_path)        
        break;
    end
    pre_img = imread(pre_path);  
    fi = pre_img;
    fi(fi<50)=0;
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
            if grey_value<125 && grey_value>75
                    pre_num = pre_num + 1 ;
                result = cat(1,result,[X,Y,dis,grey_value]);            
            end
        end
    if all(max(result(:))~=0) 
            final_pre_result = cat(1,final_pre_result,result);
    end
    end
end