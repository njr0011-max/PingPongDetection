clear all;
close all;
clc;

% -------------------------
% USER SETTINGS
% -------------------------
frame_prev = 202;
frame_curr = 203;

video_file = 'UAH_ball_video_1080p.mp4';
csv_file = 'UAH_ball_tracking_results.csv';

% If YOLO frame numbers start at 0, set this to true.
% If YOLO frame numbers start at 1, set this to false.
yolo_starts_at_zero = false;

% -------------------------
% LOAD ORIGINAL MP4 FRAME
% -------------------------
vid = VideoReader(video_file);

if yolo_starts_at_zero
    matlab_frame_prev = frame_prev + 1;
    matlab_frame_curr = frame_curr + 1;
else
    matlab_frame_prev = frame_prev;
    matlab_frame_curr = frame_curr;
end

I_prev = read(vid, matlab_frame_prev);
I_curr = read(vid, matlab_frame_curr);

% -------------------------
% LOAD YOLO CSV
% -------------------------
data = readmatrix(csv_file);

% CSV format:
% frame, ID, X, Y, width, height, velocity, confidence

row_prev = data(data(:,1) == frame_prev, :);
row_curr = data(data(:,1) == frame_curr, :);

if isempty(row_prev)
    error('No YOLO detection found for previous frame %d.', frame_prev);
end

if isempty(row_curr)
    error('No YOLO detection found for current frame %d.', frame_curr);
end

% Pick highest confidence detection if there are multiple boxes
[~, idx_prev] = max(row_prev(:,8));
[~, idx_curr] = max(row_curr(:,8));

row_prev = row_prev(idx_prev, :);
row_curr = row_curr(idx_curr, :);

% Extract YOLO boxes
% Your format is [center_x, center_y, width, height]
bbox_prev = row_prev(1, 3:6);
bbox_curr = row_curr(1, 3:6);

% -------------------------
% CONVERT CENTER BOX TO MATLAB RECTANGLE BOX
% -------------------------
cx_prev = bbox_prev(1);
cy_prev = bbox_prev(2);
w_prev  = bbox_prev(3);
h_prev  = bbox_prev(4);

cx_curr = bbox_curr(1);
cy_curr = bbox_curr(2);
w_curr  = bbox_curr(3);
h_curr  = bbox_curr(4);

% MATLAB rectangle wants [upper_left_x, upper_left_y, width, height]
bbox_prev_draw = [cx_prev - w_prev/2, cy_prev - h_prev/2, w_prev, h_prev];
bbox_curr_draw = [cx_curr - w_curr/2, cy_curr - h_curr/2, w_curr, h_curr];

% -------------------------
% MOTION VECTOR
% -------------------------
mv_x = cx_curr - cx_prev;
mv_y = cy_curr - cy_prev;

fprintf('Previous center: (%.2f, %.2f)\n', cx_prev, cy_prev);
fprintf('Current center:  (%.2f, %.2f)\n', cx_curr, cy_curr);
fprintf('Motion vector:   dx = %.2f, dy = %.2f pixels/frame\n', mv_x, mv_y);

% -------------------------
% PLOT
% -------------------------
figure('Position', [100, 100, 1200, 700]);
imshow(I_curr);
hold on;

h1 = rectangle('Position', bbox_prev_draw, ...
    'EdgeColor', 'b', 'LineWidth', 2);

h2 = rectangle('Position', bbox_curr_draw, ...
    'EdgeColor', 'r', 'LineWidth', 2);

h3 = plot(cx_prev, cy_prev, 'bo', ...
    'MarkerFaceColor', 'b', 'MarkerSize', 8);

h4 = plot(cx_curr, cy_curr, 'ro', ...
    'MarkerFaceColor', 'r', 'MarkerSize', 8);

h5 = quiver(cx_prev, cy_prev, mv_x, mv_y, 0, ...
    'g', 'LineWidth', 2, 'MaxHeadSize', 2);

title('Ball Motion Vector from YOLO Bounding Boxes');

% Dummy objects for legend only
L1 = plot(nan, nan, 'b-', 'LineWidth', 2);
L2 = plot(nan, nan, 'r-', 'LineWidth', 2);
L3 = plot(nan, nan, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
L4 = plot(nan, nan, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
L5 = plot(nan, nan, 'g-', 'LineWidth', 2);

legend([L1 L2 L3 L4 L5], ...
    {'Previous Box', 'Current Box', 'Previous Center', 'Current Center', 'Motion Vector'}, ...
    'Location', 'southwest');

hold off;
drawnow;