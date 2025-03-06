% Create Stimulation Patterns
% 16--电极数  
% 1--2D  
% [0 1]--注入方式（相邻）。[x y]:第一个模式是x,y，下一个模式是x+1,y+1
% [0 1]--测量模式。[x y]:第一个模式是x,y，下一个模式是x+1,y+1
% options：选项的单元格数组
% amplitude：驱动当前电平，默认= 0.010安培
fmdl= ng_mk_ellip_models([0,6,6,0.1],[0;45;90;135;180;225;270;315],[0.2,0,0.03]); 
background = 1.0;
[stim,msel] = mk_stim_patterns(8,1,[0,1],[0,1],{},0.01);
imdl = mk_common_model('l2c2',8);
imdl.fwd_model = fmdl;
imdl.fwd_model.stimulation = stim;
imdl.fwd_model.meas_select = msel;
imdl.hyperparameter.value = 0.001;
img= mk_image(imdl.fwd_model, background);
vh= fwd_solve( img );vh = vh.meas; 



    % 创建异常矩形区域
while(1)

    
    rec1_l = (rand()*1) + 1.4;
    rec1_w = (rand()*1) + 1.4;
    rec2_l = (rand()*1) + 1.4;
    rec2_w = (rand()*1) + 1.4;

    rect_x1 = 12 * rand() - 6;
    rect_y1 = 12 * rand() - 6;
    rect_x2 = rect_x1 + rec1_l;
    rect_y2 = rect_y1 + rec1_w;
     
    rect2_x1 = 12 * rand() - 6;
    rect2_y1 = 12 * rand() - 6;
    rect2_x2 = rect2_x1 + rec2_l;
    rect2_y2 = rect2_y1 + rec2_w;
    limte_dis = 6 - sqrt(rec1_l * rec1_l + rec1_w * rec1_w)/2 - 0.2;
    limte_dis2 = 6 - sqrt(rec2_l * rec2_l + rec2_w * rec2_w)/2 - 0.2;

    Rec_centerx = (rect_x1 + rect_x2)/2;
    Rec_centery = (rect_y1 + rect_y2)/2;

    Rec2_centerx = (rect2_x1 + rect2_x2)/2;
    Rec2_centery = (rect2_y1 + rect2_y2)/2;
    
    dis1 = distance([Rec_centerx,Rec_centery],[0,0]);
    dis2 = distance([Rec2_centerx,Rec2_centery],[0,0]);
 
    dis9 = distance([Rec_centerx,0],[Rec2_centerx,0]); %X轴距离
    dis10 = distance([0,Rec_centery],[0,Rec2_centery]); %Y轴距离
    if dis1 < limte_dis && dis2 < limte_dis2
        if dis9 > (rec1_l/2+rec2_l/2) && dis10 > (rec1_w/2+rec2_w/2)
              break
        end 
    end
end

% 随机设置异常区域的电导率，范围：3 ~ 4（背景电导率为1）
Rec_data = rand() + 3;
Rec2_data = rand() + 3;

img.elem_data(:,:) = background;
% 设置异常区域电导率
for i=1:size(img.fwd_model.elems,1)  
    x = (img.fwd_model.nodes(img.fwd_model.elems(i,1),1) + img.fwd_model.nodes(img.fwd_model.elems(i,2),1) + img.fwd_model.nodes(img.fwd_model.elems(i,3),1) ) / 3;
    y = (img.fwd_model.nodes(img.fwd_model.elems(i,1),2) + img.fwd_model.nodes(img.fwd_model.elems(i,2),2) + img.fwd_model.nodes(img.fwd_model.elems(i,3),2) ) / 3;

    if (x < rect_x2) && (x > rect_x1) && (y < rect_y2) && (y > rect_y1)                             
        img.elem_data(i) = Rec_data;
    elseif (x < rect2_x2) && (x > rect2_x1) && (y < rect2_y2) && (y > rect2_y1)
        img.elem_data(i) = Rec2_data;
    end       

end

vi = fwd_solve(img); vi = vi.meas;  
img2= inv_solve(imdl, vh, vi);
subplot(121); 
ax = gca; % 获取当前坐标轴
ax.Visible = 'off'; % 设置坐标轴不可见
show_fem(img);
subplot(122);
show_fem(img2);


coord1 = [rect_x1;rect_y1;rect_x2;rect_y2;Rec_data];
coord2 = [rect2_x1;rect2_y1;rect2_x2;rect2_y2;Rec2_data];
coord = [coord1,coord2];
name_s = num2str(1);
name_s2 = num2str(2);
data=[vh,vi];
head_data = {'vh','vi'};
head_data2 = {'Rec_coord1','Cir_coord2'};
end_data = table(vh,vi,'VariableNames',head_data);
end_data2 = table(coord1,coord2,'VariableNames',head_data2);
path_label = ['D:\',name_s,'.csv'];
path_data = ['D:\',name_s2,'.csv'];
writetable(end_data, path_data)
writetable(end_data2, path_label)










