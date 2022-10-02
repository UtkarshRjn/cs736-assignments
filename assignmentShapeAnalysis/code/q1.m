clc;
clear;
close all;
rng(42);
load('../data/ellipses2D.mat');
%Getting preshape pointset for code11
center = sum(pointSets, 2)/numOfPoints;
centroids = repmat(center, 1, numOfPoints, 1);
locStandardizedPointSet = pointSets-centroids;
norm = sqrt(sum(sum(locStandardizedPointSet.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
preshapePoints = locStandardizedPointSet./norms;

f = figure();
ax = axes(f);
hold on;
axis equal;
for i = 1:numOfPointSets
    plot(pointSets(1,:,i),pointSets(2,:,i),'LineWidth',1,'Color',[rand rand rand],'Parent',ax);
    plot([pointSets(1,32,i) pointSets(1,1,i)],[pointSets(2,32,i) pointSets(2,1,i)],'LineWidth', 1, 'Color', [rand rand rand], 'Parent', ax);
end
title('Initial Data', 'Parent', ax);
hold off;
% Code 1
initInd = randi(numOfPointSets);
mean = preshapePoints(:,:,initInd);
prevMean = mean;
i = 1;
max_iters = 300;
thrs = 5e-7;
d = 1;
while d > thrs && i < max_iters % braking condition is to check if iterations have exceeded max_iters or change in mean is very slow
    for i = 1:numOfPointSets
        rot = find_rot(mean, preshapePoints(:,:,i));
        preshapePoints(:,:,i) = rot * preshapePoints(:,:,i);
    end
    mean = sum(preshapePoints, 3)/numOfPointSets;
    norm = sqrt(sum(sum(mean.^2, 2), 1));
    mean = mean/norm;
    d = sqrt(sum(sum((mean - prevMean).^2, 2), 1));
    prevMean = mean;
    i = i+1;
end

f1 = figure();
ax1 = axes(f1);
hold on;
axis equal;
for i = 1:numOfPointSets
   plot(preshapePoints(1,:,i),preshapePoints(2,:,i),'LineWidth',1,'Color',[rand rand rand],'Parent',ax1);
   plot([preshapePoints(1,32,i) preshapePoints(1,1,i)],[preshapePoints(2,32,i) preshapePoints(2,1,i)],'LineWidth', 1, 'Color', [rand rand rand], 'Parent', ax1);
end
plot(mean(1,:),mean(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',ax1);
plot([mean(1,32) mean(1,1)],[mean(2,32) mean(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', ax1);
title('Mean Aligned Point Set using Code11');
xrange = xlim;
yrange = ylim;
hold off;

%Finding the covariance matrix
cov_matrix = zeros(64, 64);
for i = 1:numOfPointSets
    var = preshapePoints(:,:,i) - mean;
    var = reshape(var,[64 1]);
    cov_matrix = cov_matrix + var * var';
end
cov_matrix = cov_matrix / numOfPointSets;
[V, D] = eig(cov_matrix);

eig_values = diag(D);

eig_vec1 = reshape(V(:,64),[2,32]);

f2 = figure();
p1 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean - 3*sqrt(eig_values(64))*eig_vec1;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p1);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p1);
title('3 standard deviations behind on largest mode of variation','Parent',p1);
xlim(xrange);
ylim(yrange);
hold off;
p2 = subplot(1,3,2);
hold on;
axis equal;
plot(mean(1,:),mean(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p2);
plot([mean(1,32) mean(1,1)],[mean(2,32) mean(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p2);
title('Mean ellipse using code 11', 'Parent', p2);
xlim(xrange);
ylim(yrange);
hold off;
p3 = subplot(1,3,3);
hold on;
axis equal;
var2 = mean + 3*sqrt(eig_values(64))*eig_vec1;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p3);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p3);
title('3 standard deviations ahead on largest mode of variation','Parent',p3);
xlim(xrange);
ylim(yrange);
hold off;
f3 = figure();
eig_vec2 = reshape(V(:,63),[2,32]);
p11 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean - 3*sqrt(eig_values(63))*eig_vec2;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p11);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p11);
title('3 standard deviations behind on second largest mode of variation','Parent',p11);
xlim(xrange);
ylim(yrange);
hold off;
p12 = subplot(1,3,2);
hold on;
axis equal;
plot(mean(1,:),mean(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p12);
plot([mean(1,32) mean(1,1)],[mean(2,32) mean(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p12);
title('Mean ellipse using code 11', 'Parent', p12);
xlim(xrange);
ylim(yrange);
hold off;
p13 = subplot(1,3,3);
hold on;
axis equal;
var2 = mean + 3*sqrt(eig_values(63))*eig_vec2;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p13);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p13);
title('3 standard deviations ahead on second largest mode of variation','Parent',p13);
xlim(xrange);
ylim(yrange);
hold off;
f4 = figure();
eig_vec3 = reshape(V(:,62),[2,32]);
p21 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean - 3*sqrt(eig_values(62))*eig_vec3;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p21);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p21);
title('3 standard deviations behind on third largest mode of variation','Parent',p21);
xlim(xrange);
ylim(yrange);
hold off;
p22 = subplot(1,3,2);
hold on;
axis equal;
plot(mean(1,:),mean(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p22);
plot([mean(1,32) mean(1,1)],[mean(2,32) mean(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p22);
title('Mean ellipse using code 11', 'Parent', p22);
xlim(xrange);
ylim(yrange);
hold off;
p23 = subplot(1,3,3);
hold on;
var2 = mean + 3*sqrt(eig_values(62))*eig_vec3;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
axis equal;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p23);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p23);
xlim(xrange);
ylim(yrange);
title('3 standard deviations ahead on third largest mode of variation','Parent',p23);
hold off;
f5 = figure();
tom = linspace(1,64,64);
plot(tom,eig_values(64:-1:1));
title('Eigenvalues');

%Code 2

initInd = randi(numOfPointSets);
mean1 = pointSets(:,:,initInd);
mean1 = mean1 - sum(mean1, 2)/numOfPoints;
mean1 = mean1/sqrt(sum(sum(mean1.^2,2),1));
prevMean = mean1;
i = 1;
max_iters = 200;
thrs = 5e-7;
d = 1;

while d > thrs && i < max_iters % braking condition is to check if iterations have exceeded max_iters or change in mean is very slow
    temp = zeros(2,32,300);
    for i = 1:numOfPointSets
        [Mtheta, S, tx, ty] = joint(mean1, pointSets(:,:,i));
        pointSets(:,:,i) = S * Mtheta * pointSets(:,:,i) + [tx; ty];
        %temp(:,:,i) = S * Mtheta * pointSets(:,:,i) + [tx; ty];
    end
    mean1 = sum(pointSets, 3)/numOfPointSets;
    mean1 = mean1/sqrt(sum(sum(mean1.^2,2),1));
    d = sqrt(sum(sum((mean1 - prevMean).^2, 2), 1));
    prevMean = mean1;
    i = i+1;
end

f11 = figure();
ax11 = axes(f11);
hold on;
axis equal;
for i = 1:numOfPointSets
   pointSets(:,:,i) = pointSets(:,:,i) - sum(pointSets(:,:,i),2)/numOfPoints;
   pointSets(:,:,i) = pointSets(:,:,i)/sqrt(sum(sum(pointSets(:,:,i).^2,2),1));
   scatter(pointSets(1,:,i),pointSets(2,:,i),'MarkerFaceColor',[rand rand rand],'Parent',ax11);
end
plot(mean1(1,:),mean1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',ax11);
plot([mean1(1,32) mean1(1,1)],[mean1(2,32) mean1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', ax11);
title('Mean Aligned Point Set using Code22');
xrange = xlim;
yrange = ylim;
hold off;


cov_matrix = zeros(64, 64);
for i = 1:numOfPointSets
    var = pointSets(:,:,i) - mean1;
    var = reshape(var,[64 1]);
    cov_matrix = cov_matrix + var * var';
end
cov_matrix = cov_matrix / numOfPointSets;
[V, D] = eig(cov_matrix);

eig_values = diag(D);

eig_vec1 = reshape(V(:,64),[2,32]);

f12 = figure();
p31 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean1 - 3*sqrt(eig_values(64))*eig_vec1;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p31);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p31);
title('3 standard deviations behind on largest mode of variation','Parent',p31);
xlim(xrange);
ylim(yrange);
hold off;
p32 = subplot(1,3,2);
hold on;
axis equal;
plot(mean1(1,:),mean1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p32);
plot([mean1(1,32) mean1(1,1)],[mean1(2,32) mean1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p32);
title('Mean ellipse using code 22', 'Parent', p32);
xlim(xrange);
ylim(yrange);
hold off;
p33 = subplot(1,3,3);
hold on;
axis equal;
var2 = mean1 + 3*sqrt(eig_values(64))*eig_vec1;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p33);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p33);
title('3 standard deviations ahead on largest mode of variation','Parent',p33);
xlim(xrange);
ylim(yrange);
hold off;
f55 = figure();
eig_vec2 = reshape(V(:,63),[2,32]);
p41 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean1 - 3*sqrt(eig_values(63))*eig_vec2;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p41);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p41);
title('3 standard deviations behind on second largest mode of variation','Parent',p41);
xlim(xrange);
ylim(yrange);
hold off;
p42 = subplot(1,3,2);
hold on;
axis equal;
plot(mean1(1,:),mean1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p42);
plot([mean1(1,32) mean1(1,1)],[mean1(2,32) mean1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p42);
title('Mean ellipse using code 22', 'Parent', p42);
xlim(xrange);
ylim(yrange);
hold off;
p43 = subplot(1,3,3);
hold on;
axis equal;
var2 = mean1 + 3*sqrt(eig_values(63))*eig_vec2;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p43);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p43);
title('3 standard deviations ahead on second largest mode of variation','Parent',p43);
xlim(xrange);
ylim(yrange);
hold off;
f14 = figure();
eig_vec3 = reshape(V(:,62),[2,32]);
p51 = subplot(1,3,1);
hold on;
axis equal;
var1 = mean1 - 3*sqrt(eig_values(62))*eig_vec3;
norm = sqrt(sum(sum(var1.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var1 = var1./norms;
plot(var1(1,:),var1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p51);
plot([var1(1,32) var1(1,1)],[var1(2,32) var1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p51);
title('3 standard deviations behind on third largest mode of variation','Parent',p51);
xlim(xrange);
ylim(yrange);
hold off;
p52 = subplot(1,3,2);
hold on;
axis equal;
plot(mean1(1,:),mean1(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p52);
plot([mean1(1,32) mean1(1,1)],[mean1(2,32) mean1(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p52);
title('Mean ellipse using code 22', 'Parent', p52);
xlim(xrange);
ylim(yrange);
hold off;
p53 = subplot(1,3,3);
hold on;
var2 = mean1 + 3*sqrt(eig_values(62))*eig_vec3;
norm = sqrt(sum(sum(var2.^2, 2), 1));
norms = repmat(norm, 2, 32, 1);
var2 = var2./norms;
axis equal;
plot(var2(1,:),var2(2,:),'LineWidth',4,'Color',[0 0 0],'Parent',p53);
plot([var2(1,32) var2(1,1)],[var2(2,32) var2(2,1)],'LineWidth', 4, 'Color', [0 0 0], 'Parent', p53);
xlim(xrange);
ylim(yrange);
title('3 standard deviations ahead on third largest mode of variation','Parent',p53);
hold off;
f15 = figure();
tom = linspace(1,64,64);
plot(tom,eig_values(64:-1:1));
title('Eigenvalues');



function [Mtheta,S,tx,ty] = joint(X,Y)
    X1 = sum(X(1,:));
    X2 = sum(X(2,:));
    Y1 = sum(Y(1,:));
    Y2 = sum(Y(2,:));
    Z = sum(Y(2,:).^2 + X(2,:).^2);
    [~,W] = size(Y);
    C1 = sum(X(1,:).*X(2,:) + Y(1,:).*Y(2,:));
    C2 = sum(Y(1,:).*X(2,:) - X(1,:).*Y(2,:));
    A = [X2 -Y2 W 0; Y2 X2 0 W; Z 0 X2 Y2; 0 Z -Y2 X2];
    b = [X1;Y1;C1;C2];
    x = A\b;
    R = atan2(x(2), x(1));
    Mtheta = [cos(R) -sin(R); sin(R) cos(R)];
    S = x(1).^2 + x(2).^2;
    tx = x(3);
    ty = x(4);
end

function [R] = find_rot(X, Y)
    M = X*Y';
    [U,~,V] = svd(M);
    if det(V*U') == 1
        R = V*U';
    else
        sz = size(U);
        a = eye(sz(1));
        a(sz(1),sz(1)) = -1;
        R = V*(a*U');
    end
end
