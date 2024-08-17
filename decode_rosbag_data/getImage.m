path = "zed_cali.bag";
bag = rosbag(path);

imageBag = select(bag,'Topic','/zed/zed_node/left/image_rect_color');
%pcBag = select(bag,'Topic','/cloud');

imageMsgs = readMessages(imageBag);
%pcMsgs = readMessages(pcBag);

ts1 = timeseries(imageBag);
%ts2 = timeseries(pcBag);
t1 = ts1.Time;
%t2 = ts2.Time;

k = 1;
count = 0;
for i = 1:size(t1,1)
    if mod(count, 4) == 0
        idx(k) = i;
        count = count + 1;
        k = k+1;
    else
        count = count+1;
    end
end



imageFilesPath = fullfile("zed",'Images');
if ~exist(imageFilesPath,'dir')
    mkdir(imageFilesPath);
end

for i = 1:length(idx)
    I = readImage(imageMsgs{idx(i)});
    n_strPadded = sprintf('%04d',i) ;
    imageFileName = strcat(imageFilesPath,'/',n_strPadded,'.png');
    imwrite(I,imageFileName);
end

