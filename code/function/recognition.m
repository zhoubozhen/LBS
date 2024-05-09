function [result,label] = recognition(input)
% =========================================================================
% .
% -------------------------------------------------------------------------
% Input:    -  input  : The reconstructed image .
% Output:   -  result : Not labeled.
%           -  label  : The image marked by several different integers.
% =========================================================================
canny_result = edge(input,"canny",[0.3 0.6],4); %canny algorithm，0.3 and 0.6 is the threshold values, 4 is the standard deviation
se = strel('disk',4);   
bi = imclose(canny_result,se);
tianchong = imfill(bi,"holes"); % fill the hole
minSize = 40;
deleted = bwareaopen(tianchong,minSize);
[B,L] = bwboundaries(deleted,"noholes");
stats = regionprops(L,"Area","Circularity","Centroid","MajorAxisLength","MinorAxisLength","PixelList");
threshold = 0.9;
for k = 1:length(B)
  pixels = stats(k).PixelList;
  circ_value = stats(k).Circularity; % get the circularity
  diameters = mean([stats(k).MajorAxisLength stats(k).MinorAxisLength],2);
  radii = diameters/2; % radius
  % The circularity should be over 0.9 and the radius should be over 8.5, to be counted correct particles.
  if circ_value<threshold || radii>8.5  
      for i=1:size(pixels,1)
          L(pixels(i,2),pixels(i,1))=0;
      end
  end
end
inter = L;
inter(L~=0)=1;
result = deleted;
[~,new_L] = bwboundaries(inter,"noholes");
label = new_L;
end