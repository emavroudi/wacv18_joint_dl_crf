function [Y,labels] = read_data(task,subj,trial,rootfolder)
% concatenate all the time series data from the training or test sets
for i=1:numel(subj)
    for j=1:numel(trial)
        clear filename
        if (subj{i}=='H' && trial(j)==2 && strcmp(task,'Suturing')) || ...
                (subj{i}=='B' && trial(j)==5 && strcmp(task,'Knot_Tying')) || ...
                (subj{i}=='H' && trial(j)==1 && strcmp(task,'Knot_Tying')) || ...
                (subj{i}=='H' && trial(j)==2 && strcmp(task,'Knot_Tying')) || ...
                (subj{i}=='I' && trial(j)==4 && strcmp(task,'Knot_Tying')) || ...
                (subj{i}=='B' && trial(j)==5 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='E' && trial(j)==2 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='F' && trial(j)==2 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='F' && trial(j)==5 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='G' && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='H' && trial(j)==1 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='H' && trial(j)==3 && strcmp(task,'Needle_Passing')) || ...
                (subj{i}=='I' && trial(j)==1 && strcmp(task,'Needle_Passing'))
            
            Y{i,j}=[];
            labels{i,j}=[];
        else
            
            filename = [task,'_',subj{i},'00',num2str(trial(j)),'.txt'];
            rootfolder_labels = [rootfolder,'/transcriptions/'];
            rootfolder_data = [rootfolder,'/kinematics/AllGestures/'];
            data_p = 76;
            
            fileID_labels = fopen([rootfolder_labels,filename]);
            labels_temp = textscan(fileID_labels,'%f %f G%f');
            fclose(fileID_labels);
            
            fileID = fopen([rootfolder_data,filename]);
            gesture_data = textscan(fileID,'%f');
            fclose(fileID);
            gesture_data = gesture_data{:};
            
            gesture_data = reshape(gesture_data,[data_p,numel(gesture_data)/data_p]);
            G = [];
            Y_temp = [];
            for k=1:numel(labels_temp{1})
                G = [G,(labels_temp{3}(k))*(ones(1,numel(labels_temp{1}(k):labels_temp{2}(k))))];
                Y_temp = [Y_temp,gesture_data(:,labels_temp{1}(k):labels_temp{2}(k))];
            end
            labels{i,j} = G;
            Y{i,j} = Y_temp;
            
        end
    end
end

end
