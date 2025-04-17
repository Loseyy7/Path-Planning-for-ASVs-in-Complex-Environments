img = imread('astarscn1.png');
figure; imshow(img); title('Original image');

[height, width, ~] = size(img);
exclude_region = false(height, width);
exclude_start_x = width * 0.7;
exclude_start_y = 1;
exclude_width = width - exclude_start_x;
exclude_height = height * 0.3;
exclude_region(exclude_start_y:exclude_start_y+exclude_height, ...
               exclude_start_x:end) = true;

blue_channel = img(:,:,3);
red_channel = img(:,:,1);
blue_mask = (blue_channel > 100) & (red_channel < 100);
blue_mask(exclude_region) = false;

blue_mask = bwareaopen(blue_mask, 50);
blue_mask = imclose(blue_mask, strel('disk', 3));

skeleton = bwmorph(blue_mask, 'skel', Inf);

[y_coords, x_coords] = find(skeleton);

if isempty(x_coords)
    error('No path was detected, please try to adjust the color threshold or exclude area parameters.');
end

% --- bwtraceboundary ---
start_point = [y_coords(1), x_coords(1)];
boundary = bwtraceboundary(skeleton, start_point, 'N', 8, Inf, 'clockwise');
if isempty(boundary)
    error('Unable to trace the path, please check the connectivity of skeleton diagram.');
end
x_full = boundary(:,2);
y_full = boundary(:,1);


dists = sqrt(diff(x_full).^2 + diff(y_full).^2);
cumdist = [0; cumsum(dists)];

num_points = 5000;
uniform_dist = linspace(0, cumdist(end), num_points);
x = interp1(cumdist, x_full, uniform_dist);
y = interp1(cumdist, y_full, uniform_dist);

x = (x - min(x)) * 10 / (max(x) - min(x));
y = (y - min(y)) * 10 / (max(y) - min(y));

figure;
subplot(1,2,1);
imshow(img); hold on;
plot(x*(max(x_full)-min(x_full))/10+min(x_full), ...
     y*(max(y_full)-min(y_full))/10+min(y_full), 'r-', 'LineWidth', 2);
title('Original image superposition path');

subplot(1,2,2);
plot(x, y, 'b-o');
title('The extracted path coordinates');
xlabel('X'); ylabel('Y');
axis equal; grid on;

if length(x) < 3
    error('Too few path points, at least 3 points are required.');
end

angles = zeros(1, length(x)-2);
for i = 1:length(x)-2
    v1 = [x(i+1)-x(i), y(i+1)-y(i)];
    v2 = [x(i+2)-x(i+1), y(i+2)-y(i+1)];
    angles(i) = acosd(dot(v1,v2)/(norm(v1)*norm(v2)));
end

dx = gradient(x);
dy = gradient(y);
denominator = (dx.^2 + dy.^2).^(3/2);
denominator(denominator == 0) = eps;
curvature = (dx.*gradient(dy) - dy.*gradient(dx)) ./ denominator;

fprintf('=== Path analysis results ===\n');
fprintf('Number of points in the path: %d\n', length(x));
fprintf('Total angle change: %.2f°\n', sum(angles));
fprintf('Average angle change: %.2f°/segment\n', mean(angles));
fprintf('Maximum angle change: %.2f°\n', max(angles));
fprintf('Mean curvature: %.4f\n', mean(abs(curvature)));
fprintf('Curvature standard deviation: %.4f\n', std(curvature));
