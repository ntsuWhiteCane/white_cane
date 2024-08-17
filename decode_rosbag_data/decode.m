clear;
path = "data_1.bag";
bag = rosbag(path);

n = 4;
imageBag = select(bag,'Topic','/zed/zed_node/left/image_rect_color');
zedPointCloudBag = select(bag,'Topic','/zed/zed_node/point_cloud/cloud_registered');
pcBag = select(bag,'Topic','/cloud');
%depthBag = select(bag, 'Topic', '/zed/zed_node/depth/depth_registered');
pc2Bag = select(bag,'Topic','/lidar2/cloud');

imageMsgs = readMessages(imageBag);
zedPointCloudMsgs = readMessages(zedPointCloudBag);
pcMsgs = readMessages(pcBag);
pc2Msgs = readMessages(pc2Bag);
%depth = readMessages(depthBag, 'DataFormat', 'struct');

ts1 = timeseries(imageBag);
ts2 = timeseries(pcBag);
ts3 = timeseries(zedPointCloudBag);
%ts4 = timeseries(depthBag);
ts4 = timeseries(pc2Bag);

t1 = ts1.Time;
t2 = ts2.Time;
t3 = ts3.Time;
t4 = ts4.Time;
timeArray = {t1, t2, t3, t4};
k = 1;
count = 0;
ss = [size(timeArray{1}, 1), size(timeArray{2}, 1), size(timeArray{3}, 1), size(timeArray{4}, 1)];
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

pcFilesPath = fullfile("data",'PointClouds');
imageFilesPath = fullfile("data",'Images');
zedPointCloudFilePath = fullfile("data", "Zed_Point_Clouds");
depthFilePath = fullfile("data", "Depth");
pc2FilePath = fullfile("data", "PointClouds2");

if ~exist(imageFilesPath,'dir')
    mkdir(imageFilesPath);
end
if ~exist(pcFilesPath,'dir')
    mkdir(pcFilesPath);
end
if ~exist(zedPointCloudFilePath, 'dir')
    mkdir(zedPointCloudFilePath);
end
if ~exist(depthFilePath, 'dir')
    mkdir(depthFilePath);
end
if ~exist(pc2FilePath, 'dir')
    mkdir(pc2FilePath);
end

for i = 1:length(idx)
    I = readImage(imageMsgs{idx(i,1)});
    zed_pc = pointCloud(readXYZ(zedPointCloudMsgs{idx(i, 3)}));
    pc = pointCloud(readXYZ(pcMsgs{idx(i,2)}));
    pc2 = pointCloud(readXYZ(pc2Msgs{idx(i, 4)}));
    %depth_u32 = reshape(typecast(depth{idx(i, 4)}.Data, 'single'), 360, 640);

    n_strPadded = sprintf('%04d',i) ;
    pcFileName = strcat(pcFilesPath,'/',n_strPadded,'.pcd');
    zedPointCloudFileName = strcat(zedPointCloudFilePath, '/', n_strPadded, '.pcd');
    imageFileName = strcat(imageFilesPath,'/',n_strPadded,'.png');
    depthFileName = strcat(depthFilePath, '/', n_strPadded, '.mat');
    pc2FileName = strcat(pc2FilePath, '/', n_strPadded, ".pcd");

    imwrite(I,imageFileName);
    pcwrite(pc,pcFileName);
    pcwrite(zed_pc, zedPointCloudFileName);
    pcwrite(pc2, pc2FileName);
    %save(depthFileName, "depth_u32");
end
