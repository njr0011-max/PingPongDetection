%YOLO_box_Motion_Vector
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

% Block matching settings
BS = 10;          % block size
SW = 100;          % search window
ROI_PAD = 50;    % extra pixels around YOLO ball box

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

I_prev_rgb = read(vid, matlab_frame_prev);
I_curr_rgb = read(vid, matlab_frame_curr);

% Convert to grayscale for SAD block matching
I_prev = rgb2gray(I_prev_rgb);
I_curr = rgb2gray(I_curr_rgb);

I_prev = im2double(I_prev);
I_curr = im2double(I_curr);

[y_dim, x_dim] = size(I_curr);

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

bbox_prev_draw = [cx_prev - w_prev/2, cy_prev - h_prev/2, w_prev, h_prev];
bbox_curr_draw = [cx_curr - w_curr/2, cy_curr - h_curr/2, w_curr, h_curr];

fprintf('Previous center from YOLO: (%.2f, %.2f)\n', cx_prev, cy_prev);
fprintf('Current center from YOLO:  (%.2f, %.2f)\n', cx_curr, cy_curr);

% -------------------------
% MAKE ROI AROUND YOLO BOXES
% -------------------------
% Use both previous and current boxes so the ROI covers the ball movement.

x1_prev = bbox_prev_draw(1);
y1_prev = bbox_prev_draw(2);
x2_prev = bbox_prev_draw(1) + bbox_prev_draw(3);
y2_prev = bbox_prev_draw(2) + bbox_prev_draw(4);

x1_curr = bbox_curr_draw(1);
y1_curr = bbox_curr_draw(2);
x2_curr = bbox_curr_draw(1) + bbox_curr_draw(3);
y2_curr = bbox_curr_draw(2) + bbox_curr_draw(4);

roi_x1 = floor(min(x1_prev, x1_curr) - ROI_PAD);
roi_y1 = floor(min(y1_prev, y1_curr) - ROI_PAD);
roi_x2 = ceil(max(x2_prev, x2_curr) + ROI_PAD);
roi_y2 = ceil(max(y2_prev, y2_curr) + ROI_PAD);

% Keep ROI inside image
roi_x1 = max(1, roi_x1);
roi_y1 = max(1, roi_y1);
roi_x2 = min(x_dim, roi_x2);
roi_y2 = min(y_dim, roi_y2);

fprintf('ROI x range: %d to %d\n', roi_x1, roi_x2);
fprintf('ROI y range: %d to %d\n', roi_y1, roi_y2);

% -------------------------
% BLOCK MATCHING ONLY INSIDE ROI
% -------------------------
arrow_x = [];
arrow_y = [];
arrow_u = [];
arrow_v = [];
sad_list = [];

for y = roi_y1:BS:(roi_y2 - BS + 1)
    for x = roi_x1:BS:(roi_x2 - BS + 1)

        % Current block
        curr_blk = I_curr(y:y+BS-1, x:x+BS-1);

        SAD_min = inf;
        dy_min = 0;
        dx_min = 0;

        % Search nearby blocks in previous frame
        for dy = -SW:SW
            for dx = -SW:SW

                UL_y = y + dy;
                UL_x = x + dx;

                LR_y = UL_y + BS - 1;
                LR_x = UL_x + BS - 1;

                if UL_y < 1 || UL_x < 1
                    continue;
                end

                if LR_y > y_dim || LR_x > x_dim
                    continue;
                end

                candi_blk = I_prev(UL_y:LR_y, UL_x:LR_x);

                SAD = sum(sum(abs(curr_blk - candi_blk)));

                if SAD < SAD_min
                    SAD_min = SAD;
                    dy_min = dy;
                    dx_min = dx;
                end
            end
        end

        % Center of current block
        block_center_x = x + BS/2;
        block_center_y = y + BS/2;

        % Since we found where the current block came from in the previous frame,
        % reverse the sign to show motion from previous frame to current frame.
        mv_x = -dx_min;
        mv_y = -dy_min;

        arrow_x = [arrow_x; block_center_x];
        arrow_y = [arrow_y; block_center_y];
        arrow_u = [arrow_u; mv_x];
        arrow_v = [arrow_v; mv_y];
        sad_list = [sad_list; SAD_min];

    end
end

fprintf('Total block vectors computed: %d\n', length(arrow_x));

% -------------------------
% FILTER BAD / RANDOM VECTORS
% -------------------------
motion_mag = sqrt(arrow_u.^2 + arrow_v.^2);

% Remove zero vectors and extremely large vectors
good = motion_mag > 0 & motion_mag <= SW;

arrow_x = arrow_x(good);
arrow_y = arrow_y(good);
arrow_u = arrow_u(good);
arrow_v = arrow_v(good);
sad_list = sad_list(good);

fprintf('Vectors after basic filtering: %d\n', length(arrow_x));

% Optional: keep only the better matching blocks
% Lower SAD means better match.
if ~isempty(sad_list)
    sad_threshold = prctile(sad_list, 70);
    good_sad = sad_list <= sad_threshold;

    arrow_x = arrow_x(good_sad);
    arrow_y = arrow_y(good_sad);
    arrow_u = arrow_u(good_sad);
    arrow_v = arrow_v(good_sad);

    fprintf('Vectors after SAD filtering: %d\n', length(arrow_x));
end

% -------------------------
% AVERAGE MOTION VECTOR FROM BLOCKS
% -------------------------
if ~isempty(arrow_u)
    avg_mv_x = mean(arrow_u);
    avg_mv_y = mean(arrow_v);

    fprintf('Average block motion vector: dx = %.2f, dy = %.2f pixels/frame\n', avg_mv_x, avg_mv_y);
else
    avg_mv_x = 0;
    avg_mv_y = 0;
    warning('No valid block motion vectors found.');
end

% -------------------------
% PLOT
% -------------------------
figure('Position', [100, 100, 1200, 700]);
imshow(I_curr_rgb);
hold on;

% Previous YOLO box
rectangle('Position', bbox_prev_draw, ...
    'EdgeColor', 'b', 'LineWidth', 2);

% Current YOLO box
rectangle('Position', bbox_curr_draw, ...
    'EdgeColor', 'r', 'LineWidth', 2);

% ROI box
rectangle('Position', [roi_x1, roi_y1, roi_x2-roi_x1, roi_y2-roi_y1], ...
    'EdgeColor', 'y', 'LineWidth', 2);

% YOLO centers
plot(cx_prev, cy_prev, 'bo', ...
    'MarkerFaceColor', 'b', 'MarkerSize', 8);

plot(cx_curr, cy_curr, 'ro', ...
    'MarkerFaceColor', 'r', 'MarkerSize', 8);

% Many block motion vectors
quiver(arrow_x, arrow_y, arrow_u, arrow_v, 0, ...
    'g', 'LineWidth', 1.5);

% Average motion vector from YOLO current center
quiver(cx_curr, cy_curr, avg_mv_x, avg_mv_y, 0, ...
    'm', 'LineWidth', 3, 'MaxHeadSize', 3);

title(sprintf('Block Motion Vectors in YOLO ROI: Frame %d to %d', frame_prev, frame_curr));

% Dummy legend objects
L1 = plot(nan, nan, 'b-', 'LineWidth', 2);
L2 = plot(nan, nan, 'r-', 'LineWidth', 2);
L3 = plot(nan, nan, 'y-', 'LineWidth', 2);
L4 = plot(nan, nan, 'g-', 'LineWidth', 2);
L5 = plot(nan, nan, 'm-', 'LineWidth', 3);

legend([L1 L2 L3 L4 L5], ...
    {'Previous YOLO Box', 'Current YOLO Box', 'ROI', 'Block Motion Vectors', 'Average Motion Vector'}, ...
    'Location', 'southwest');

hold off;
drawnow;