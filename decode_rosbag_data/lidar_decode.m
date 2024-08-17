clear;
path = "data_2.bag";
bag = rosbag(path);

n = 2;
pc1Bag = select(bag,'Topic','/cloud');
pc2Bag = select(bag,'Topic','/lidar2/cloud');

pc1Msgs = readMessages(pc1Bag);
pc2Msgs = readMessages(pc2Bag);

ts1 = timeseries(pc1Bag);
ts2 = timeseries(pc2Bag);

t1 = ts1.Time;
t2 = ts2.Time;
timeArray = {t1, t2};
k = 1;
count = 0;
ss = [size(timeArray{1}, 1), size(timeArray{2}, 1)];
[less_content_data, less_content_id] = min(ss);
for i = 1:size(timeArray{less_content_id}, 1)
    for j=1:n
        time2 = timeArray{less_content_id};
        time1 = timeArray{j};
        [val, indx] = min(abs(time2(i) - time1));
        tmp_val(j) = val;
        tmp_idx(j) = indx;
    end

    if(max(tmp_val) <= 0.1)
        if count < 5
            count = count + 1;
        else
            idx(k, :) = tmp_idx;
            k = k+1;
            count = 0;
        end
    end
end

pc1FilesPath = fullfile("lidar",'PointClouds1');
pc2FilesPath = fullfile("lidar",'PointClouds2');

if ~exist(pc1FilesPath,'dir')
    mkdir(pc1FilesPath);
end
if ~exist(pc2FilesPath,'dir')
    mkdir(pc2FilesPath);
end


for i = 1:length(idx)
    pc1 = pointCloud(readXYZ(pc1Msgs{idx(i,1)}));
    pc2 = pointCloud(readXYZ(pc2Msgs{idx(i,2)}));

    n_strPadded = sprintf('%04d',i) ;
    pc1FileName = strcat(pc1FilesPath,'/',n_strPadded,'.pcd');
    pc2FileName = strcat(pc2FilesPath,'/',n_strPadded,'.pcd');

    pcwrite(pc1,pc1FileName);
    pcwrite(pc2,pc2FileName);
end
