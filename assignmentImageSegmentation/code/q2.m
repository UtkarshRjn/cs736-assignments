unzip("../data/assignmentSegmentBrainGmmEmMrf.mat.zip","../data/");
data = load("../data/assignmentSegmentBrainGmmEmMrf.mat");

imageData = getfield(data,"imageData");
imageMask = getfield(data,"imageMask");

K = 3;
X = initial_label_estimate(imageData, imageMask);
mu = zeros(1,K);
sigma = zeros(1,K);
beta = 0.35;
num_iter = 25;

%% Initializing mean and sigma for our initialized labels
for i=1:K
    index = (X == i);
    mu(i) = mean(imageData(index));
    sigma(i) = std(imageData(index));
end

for i=1:num_iter
    
    % E step
    % Hard Segmentation
    log_posterior_before = compute_posterior(X,imageData,imageMask,beta,mu,sigma);
    X = modified_ICM(imageData,imageMask,X,K,beta,mu,sigma);
    log_posterior_after = compute_posterior(X,imageData,imageMask,beta,mu,sigma);
    fprintf('Iter %d: Before ICM : %.4f, After ICM :%.4f\n',i,log_posterior_before,log_posterior_after);

    % Soft Segmentation
    gamma = find_memberships(imageData, imageMask, K, X, mu, sigma, beta); % membership update
    
    %M step
    mu = find_class_mean(mu,gamma,imageData); % class mean update
    sigma = find_class_std(sigma,gamma,imageData,mu); % class std update
    
end

% images
% (i) Corrected image provided
f1 = imagesc(imageData);colormap 'gray';title('Original image');colorbar;
saveas(f1,"q2_CorruptedImage","png")

% (ii) Optimal class-membership image estimates for β = 0
f2 = imagesc(gamma(:,:,1));colormap 'gray';title('Label 1 class-membership, \beta = 0.35');colorbar;
saveas(f2,"q2_MembershipImage_1_with_mrf","png")
f3 = imagesc(gamma(:,:,2));colormap 'gray';title('Label 2 class-membership, \beta = 0.35');colorbar;
saveas(f3,"q2_MembershipImage_2_with_mrf","png")
f4 = imagesc(gamma(:,:,3));colormap 'gray';title('Label 3 class-membership, \beta = 0.35');colorbar;
saveas(f4,"q2_MembershipImage_3_with_mrf","png")

% (iii) Optimal class-membership image estimates for β = 0
f5 = imagesc(X);colormap 'gray';title('MRF-GMM-EM Segmentation, \beta = 0.35');colorbar;
saveas(f5,"q2_gmm_mrf_em_segmentation_with_beta","png")

fprintf('The optimal estimates for class means are %.4f, %.4f. %.4f\n', mu(1),mu(2),mu(3));
%% 
% 

K = 3;
X = initial_label_estimate(imageData, imageMask);
mu = zeros(1,K);
sigma = zeros(1,K);
beta = 0;
num_iter = 25;

%% Initializing mean and sigma for our initialized labels
for i=1:K
    index = (X == i);
    mu(i) = mean(imageData(index));
    sigma(i) = std(imageData(index));
end

for i=1:num_iter
    
    % E step
    % Hard Segmentation
    log_posterior_before = compute_posterior(X,imageData,imageMask,beta,mu,sigma);
    X = modified_ICM(imageData,imageMask,X,K,beta,mu,sigma);
    log_posterior_after = compute_posterior(X,imageData,imageMask,beta,mu,sigma);
    fprintf('Iter %d: Before ICM : %.4f, After ICM :%.4f\n',i,log_posterior_before,log_posterior_after);

    % Soft Segmentation
    gamma = find_memberships(imageData, imageMask, K, X, mu, sigma, beta); % membership update
    
    %M step
    mu = find_class_mean(mu,gamma,imageData); % class mean update
    sigma = find_class_std(sigma,gamma,imageData,mu); % class std update
    
end

% images

% (iv) Optimal class-membership image estimates for chosen β
f6 = imagesc(gamma(:,:,1));colormap 'gray';title('Label 1 class-membership, \beta = 0');colorbar;
saveas(f6,"q2_MembershipImage_1_without_mrf","png")
f7 = imagesc(gamma(:,:,2));colormap 'gray';title('Label 2 class-membership, \beta = 0');colorbar;
saveas(f7,"q2_MembershipImage_2_without_mrf","png")
f8 = imagesc(gamma(:,:,3));colormap 'gray';title('Label 3 class-membership, \beta = 0');colorbar;
saveas(f8,"q2_MembershipImage_3_without_mrf","png")

% (v) Optimal label image estimate for chosen β
f9 = imagesc(X);colormap 'gray';title('MRF-GMM-EM Segmentation, \beta = 0');colorbar;
saveas(f9,"q2_gmm_mrf_em_segmentation_without_beta","png")

%%
function result = compute_posterior(X,Y,M,beta,mu,sigma)
    P = 0;
    for i = 1:size(Y,1)
        for j = 1:size(Y,2)
            if (M(i, j) == 0)
                continue;
            end
            
            x = X(i,j);
            P = P +  exp(-1*(1-beta)*((X(i, j) - mu(x))^2)/(2*sigma(x)))*getPrior(X, X(i,j), i, j, M, beta);
        end 
    end
    result = log(P);
end
%% 
% 

function X = initial_label_estimate( Y, M)
    minPxl = min(Y(:));
    Y = Y - minPxl;
    maxPxl = max(Y(:));
    
    X = zeros(size(Y));
    
    pos = and(Y <= maxPxl, M == 1);
    X(pos) = 3;    
    pos = and(Y <= 2 * maxPxl / 3, M == 1);
    X(pos) = 2;
    pos = and(Y <= maxPxl / 3, M == 1);
    X(pos) = 1; 
end
%% 
% 

function new_X = modified_ICM(Y,M,X,K,beta,mu,sigma)
    for i = 1:size(Y,1)
        for j = 1:size(Y,2)
            
            if (M(i, j) == 0)
                continue;
            end
            
            Pmax_index = -1;
            Pmax = -1;
            for index = 1:K
                gauss = exp(- ( 1 - beta ) * (Y(i, j) - mu(1, index))^2 / (2 * sigma(1, index) * sigma(1, index)));
                mrf_prior = getPrior(X, index, i, j, M, beta);
                P = gauss*mrf_prior;
                if P > Pmax
                    Pmax = P;
                    Pmax_index = index;
                end
            end
            X(i,j) = Pmax_index;
        end
    end 
    new_X = X;
end
%% 
% 

function mu = find_class_mean(mu,gamma,Y)
    for i = 1:size(gamma,3)
        m = gamma(:, :, i);
        N = m .* Y;
        N = sum(N(:));
        mu(1,i) = N / sum(m(:));
    end
end
%% 
% 

function sigma = find_class_std(sigma,gamma,Y,mu)
    for i = 1:size(gamma,3)
        D = sum(sum(gamma(:, :,i)));
        N = (Y(:, :) - mu(1,i)).^2;
        N = gamma(:, :, i) .* N;
        N = sum(N(:));
        sigma(1,i) = sqrt(N / D);
    end
end
%% 
% 

function [ prior ] = getPrior( X, x, i, j, M, beta)
    diff_count = 0;
    if M(i - 1, j) == 1
        diff_count = diff_count + (X(i - 1, j) ~= x);
    end
    if M(i + 1, j) == 1
        diff_count = diff_count + (X(i + 1, j) ~= x);
    end
    if M(i, j - 1) == 1
        diff_count = diff_count + (X(i, j - 1) ~= x);
    end
    if M(i, j + 1) == 1
        diff_count = diff_count + (X(i, j + 1) ~= x);
    end

    prior = exp(-diff_count * beta);
end
%% 
% 

function [gamma] = find_memberships(Y, M, K, X, mu, sigma, beta)
    gamma = zeros(size(Y,1),size(Y,2),K);
    for r = 1:size(Y,1)
       for c = 1:size(Y,2)
           if(M(r,c) == 0)
               continue;
           end
           
           memberships = zeros(K,1);
           for x=1:K
               gauss = exp(- ( 1 - beta ) * (Y(r, c) - mu(1, x))^2 / (2 * sigma(1, x) * sigma(1, x)));
               mrf_prior = getPrior(X, x, r, c, M, beta);
               memberships(x) = gauss * mrf_prior;
           end
           
           memberships = memberships ./ sum(memberships);
           if (sum(isnan(memberships)) > 0)
               gamma(r,c,:) = gamma(r,c,:);
           else
               gamma(r,c,:) = memberships;
           end
           
       end
   end
end