clear;
path = ".\\bag\\test_310_600.bag";
bag = rosbag(path);

topic_list = ["/zed/zed_node/left/image_rect_color", "/zed/zed_node/depth/depth_registered", "/zed/zed_node/disparity/disparity_image"];
n = length(topic_list);

for i=1:n
    topic_bag{i} = select(bag, "Topic", topic_list(i));
    msgs{i} = readMessages(topic_bag{i});
    timeArray{i} = timeseries(topic_bag{i}).Time;
    ss(i) = size(timeArray{i}, 1);
end

k = 1;
count = 0;
[less_content_data, less_content_id] = min(ss);
%depth = readMessages(depthBag, 'DataFormat', 'struct');

for i = 1:size(timeArray{less_content_id}, 1)
    for j=1:n
        time2 = timeArray{less_content_id};
        time1 = timeArray{j};
        [val, indx] = min(abs(time2(i) - time1));
        tmp_val(j) = val;
        tmp_idx(j) = indx;
    end

    if(max(tmp_val) <= 0.1)
        if count < 2
            count = count + 1;
        else
            idx(k, :) = tmp_idx;
            k = k+1;
            count = 0;
        end
    end
end

pcFilesPath = fullfile("data",'PointClouds1');
pc2FilePath = fullfile("data", "PointClouds2");
imageFilesPath = fullfile("data",'Images');
zedPointCloudFilePath = fullfile("data", "Zed_Point_Clouds");
depthFilePath = fullfile("data", "Depth");
dispFilePath = fullfile("data", "Disp");
imuFilePath = fullfile("data", "Imu");

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
if ~exist(dispFilePath, "dir")
    mkdir(dispFilePath);
end

if ~exist(imuFilePath, 'dir')
    mkdir(imuFilePath);
end

for i = 1:length(idx)

    n_strPadded = sprintf('%04d',i) ;

    zed_image = readImage(msgs{1}{idx(i,1)});
    depth_u32 = msgs{2}{idx(i, 2)}.Data;
    disp_u32 = msgs{3}{idx(i, 3)}.Image.Data;

    pcFileName = strcat(pcFilesPath,'/',n_strPadded,'.pcd');
    pc2FileName = strcat(pc2FilePath, '/', n_strPadded, ".pcd");
    zedPointCloudFileName = strcat(zedPointCloudFilePath, '/', n_strPadded, '.pcd');

    imageFileName = strcat(imageFilesPath,'/',n_strPadded,'.png');
    depthFileName = strcat(depthFilePath, '/', n_strPadded, '.mat');
    dispFileName = strcat(dispFilePath, '/', n_strPadded, ".mat");
    imuFileName = strcat(imuFilePath, '/', n_strPadded, ".mat");

    imwrite(zed_image, imageFileName);
    %pcwrite(pc,pcFileName);
    %pcwrite(zed_pc, zedPointCloudFileName);
    %pcwrite(pc2, pc2FileName);
    %save(imuFileName, "ypr");
    save(depthFileName, "depth_u32");
    save(dispFileName, "disp_u32");
end
