
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


    img.elem_data(:,:) = background;
    ell_data = rand() * 0.1 + 0.3;
    for i=1:size(img.fwd_model.elems,1)  
        x = (img.fwd_model.nodes(img.fwd_model.elems(i,1),1) + img.fwd_model.nodes(img.fwd_model.elems(i,2),1) + img.fwd_model.nodes(img.fwd_model.elems(i,3),1) ) / 3;
        y = (img.fwd_model.nodes(img.fwd_model.elems(i,1),2) + img.fwd_model.nodes(img.fwd_model.elems(i,2),2) + img.fwd_model.nodes(img.fwd_model.elems(i,3),2) ) / 3;
        eli_pos1 = (y*y)/(3*3)+((x - 2)*(x - 2))/(1.4*1.4);
        eli_pos2 = (y*y)/(3*3)+((x + 2)*(x + 2))/(1.4*1.4);
    
        if eli_pos1 < 1 ||  eli_pos2 <1                            
            img.elem_data(i) = ell_data;
        end
    
    end
    while(1)
        abnormal_r1 = rand() * 0.5 + 1;
        Cir_data = rand() * 0.5 + 0.48;

        tar1_X = 12 * rand() - 6;
        tar1_Y = 12 * rand() - 6;

        dis_c1 = distance([tar1_X,tar1_Y],[0,0]);


        ww = abs(ell_data - Cir_data);
        if dis_c1 < (6 - abnormal_r1-0.2) && ww >0.05
            break
        end
    end
    
    
    target = mk_c2f_circ_mapping(img.fwd_model, [tar1_X;tar1_Y;abnormal_r1]);
    
    A1 = full(target);
  
    for i = 1 : size(A1)
        if A1(i,:) ~= 0
            A1(i,:) = Cir_data;
            img.elem_data(i,:) = A1(i,:);    
        end
    end
   
    vi = fwd_solve(img); vi = vi.meas;  
    img2= inv_solve(imdl, vh, vi);
    subplot(231); 
    show_fem(img);
    subplot(232);
    show_fem(img2);
    subplot(233);
    data = zeros(64,64);
    data(:,:) = show_slices(img);
    b = unique(data);

    for i=1:64
        for j=1:64
            if data(i,j)==b(1)
                data(i,j)=0.0;
            elseif data(i,j)==b(4)
                data(i,j)=1.0;
            else
                if ell_data > Cir_data
                    if data(i,j)==b(2)
                        data(i,j) = Cir_data;
                    else
                        data(i,j) = ell_data;
                    end
                else
                    if data(i,j)==b(2)
                        data(i,j) = ell_data;
                    else
                        data(i,j) = Cir_data;
    
                    end
                end
            end

        end
    end

name_s = num2str(1);
name_s2 = num2str(2);
measure=[vh,vi];
head_data = {'vh','vi'};
end_data = table(vh,vi,'VariableNames',head_data);
path_label = ['D:\',name_s,'.csv'];
path_data = ['D:\',name_s2,'.csv'];
writetable(end_data, path_data)    
writematrix(data,path_label);








