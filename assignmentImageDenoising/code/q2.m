%% Problem 2

data = load('../data/brainMRIslice.mat');

image_noisy = getfield(data, "brainMRIsliceNoisy");
image_noiseless = getfield(data, "brainMRIsliceOrig");

% Because the data is an MRI image we would use a Rician noise model.
initial_rrmse = RRMSE(image_noiseless,image_noisy)
Y = image_noisy;

h = surf(image_noisy);
colormap("jet");
cb = colorbar; 
ylabel(cb,'voltage') 
view([0.00 -90.00])
title("Noisy Image");
set(h,'LineStyle','None');
saveas(gcf,sprintf('noisy_colormap.png'));

h = surf(image_noiseless);
colormap("jet");
view([0.00 -90.00])
title("Noiseless Image");
cb = colorbar; 
ylabel(cb,'voltage');
set(h,'LineStyle','None');
saveas(gcf,sprintf('noiseless_colormap.png'));
%% 
% MRF prior: Quadratic function: g_1(u) = |u|^2

% (b) MRF is a qudratic function
% dg(u)/du = 2u

best_alpha = 0.94;

[denoised_image , losses] = gradientDescent(image_noisy,1,best_alpha,0);
[denoised_image2 ] = gradientDescent(image_noisy,1,best_alpha*0.8,0);
[denoised_image3 ] = gradientDescent(image_noisy,1,best_alpha*1.2,0);

plot(losses);
title("Quadratic Prior");
xlabel("Number of Iterations");
ylabel("Objective Function");
saveas(gcf,sprintf('quadratic_prior_plot.png'));

quad_rrmse = RRMSE(image_noiseless,denoised_image)
quad_rrmse_l_alpha = RRMSE(image_noiseless,denoised_image2)
quad_rrmse_h_alpha = RRMSE(image_noiseless,denoised_image3)

imshow(denoised_image);
h = surf(denoised_image);
colormap("jet");
title("Quadratic Prior");
cb = colorbar; 
view([0.00 -90.00])
ylabel(cb,'voltage') 
set(h,'LineStyle','None');
saveas(gcf,sprintf('quadratic_prior_colormap.png'));
%% 
% MRF prior: Discontinuity-adaptive Huber function:

% (c) MRF is a huber function
% dg(u)/du = u for|u| <= gamma
% dg(u)/du = -gamma * u / |u|
alpha2 = 0.25;
gamma = 0.007;

[denoised_image , losses] = gradientDescent(image_noisy,2,alpha2,gamma);
[denoised_image2 ] = gradientDescent(image_noisy,2,alpha2*0.8,gamma);
[denoised_image3 ] = gradientDescent(image_noisy,2,alpha2*1.2,gamma);
[denoised_image4 ] = gradientDescent(image_noisy,2,alpha2,gamma*0.8);
[denoised_image5 ] = gradientDescent(image_noisy,2,alpha2,gamma*1.2);

plot(losses);
title("Huber Prior");
xlabel("Number of Iterations");
ylabel("Objective Function");
saveas(gcf,sprintf('huber_prior_plot.png'));

huber_rrmse = RRMSE(image_noiseless,denoised_image)
huber_rrmse_l_alpha = RRMSE(image_noiseless,denoised_image2)
huber_rrmse_h_alpha = RRMSE(image_noiseless,denoised_image3)
huber_rrmse_l_gamma = RRMSE(image_noiseless,denoised_image4)
huber_rrmse_h_gamma = RRMSE(image_noiseless,denoised_image5)
imshow(denoised_image);
h = surf(denoised_image);
colormap("jet");
title("Huber Prior");
cb = colorbar; 
view([0.00 -90.00])
ylabel(cb,'voltage') 
set(h,'LineStyle','None');
saveas(gcf,sprintf('huber_prior_colormap.png'));
%% 
% MRF prior: Discontinuity-adaptive function:

% (d) MRF is a discontinuity adapted function
% dg(u)/du = gamma*u/(gamma+abs(u))
alpha3 = 0.48;
gamma2 = 0.065;

[denoised_image , losses] = gradientDescent(image_noisy,3,alpha3,gamma2);
[denoised_image2 ] = gradientDescent(image_noisy,3,alpha3*0.8,gamma2);
[denoised_image3 ] = gradientDescent(image_noisy,3,alpha3*1.2,gamma2);
[denoised_image4 ] = gradientDescent(image_noisy,3,alpha3,gamma2*0.8);
[denoised_image5 ] = gradientDescent(image_noisy,3,alpha3,gamma2*1.2);

plot(losses);
title("Discontinuity-Adaptive Prior");
xlabel("Number of Iterations");
ylabel("Objective Function");
saveas(gcf,sprintf('discontinuity_adaptive_prior_plot.png'));

adap_rrmse = RRMSE(image_noiseless,denoised_image)
adap_rrmse_l_alpha = RRMSE(image_noiseless,denoised_image2)
adap_rrmse_h_alpha = RRMSE(image_noiseless,denoised_image3)
adap_rrmse_l_gamma = RRMSE(image_noiseless,denoised_image4)
adap_rrmse_h_gamma = RRMSE(image_noiseless,denoised_image5)

imshow(denoised_image);
h = surf(denoised_image);
colormap("jet");
title("Discontinuity-Adaptive Prior");
cb = colorbar; 
view([0.00 -90.00])
ylabel(cb,'voltage') 
set(h,'LineStyle','None');
saveas(gcf,sprintf('discontinuity_adaptive_prior_colormap.png'));
%%
function [denoised_image, losses] = gradientDescent(imageNoisy, algorithm ,alpha ,gamma,image_noiseless)

    num_iter = 100;
    tau = 0.005;
    losses = zeros(num_iter,1);
    X = imageNoisy;
    Y = imageNoisy;
    last_obj_fun = Inf;
    
    for i=1:num_iter
        
        [mrf,der_mrf] = calc_potential(X,gamma,algorithm);  
        
    %     L = alpha1*X - alpha1*(besselj(1,alpha1*(X.*Y))./besselj(0,alpha1*(X.*Y))).*Y;
        L = alpha*2*(X-Y);
        der = (1-alpha)*der_mrf;
        dX = L - der;
        
    %     likelihood = rician(X,Y,alpha1);
        likelihood = gaussian(X,Y,alpha);
        energy = (1-alpha)*mrf;
        obj_fun = likelihood + energy;
        losses(i) = sum(obj_fun(:));
        
        new_X = X - tau*dX;
        
        if sum(obj_fun(:)) < sum(last_obj_fun(:))
            last_obj_fun = obj_fun;
            tau = 1.1 * tau;
            X = new_X;
        else
            tau = tau*0.5; 
        end
        
    end

    denoised_image = X;
end
%% 
% Function to find derivative of potential function and the potential function:

function [mrf,der_mrf] = calc_potential(X,gamma,algorithm)
    
    S = size(X);
    Z = zeros(1,S(1));
    temp_s = [diff(X) ; Z];
    temp_n = [Z; -diff(X)];
    temp_e = [diff(X,1,2),Z'];
    temp_w = [Z',-diff(X,1,2)];
    
    if algorithm==1
        der_mrf = der_quadratic(temp_w) + der_quadratic(temp_e) + der_quadratic(temp_n) + der_quadratic(temp_s);
        mrf = quadratic(temp_w) + quadratic(temp_e) + quadratic(temp_n) + quadratic(temp_s);    
    elseif algorithm==2
        der_mrf = der_huber(temp_w,gamma) + der_huber(temp_e,gamma) + der_huber(temp_n,gamma) + der_huber(temp_s,gamma);
        mrf = huber(temp_w,gamma) + huber(temp_e,gamma) + huber(temp_n,gamma) + huber(temp_s,gamma); 
    elseif algorithm==3
        der_mrf = der_disc_adap_prior(temp_w,gamma) + der_disc_adap_prior(temp_n,gamma) + der_disc_adap_prior(temp_e,gamma) + der_disc_adap_prior(temp_s,gamma);
        mrf = disc_adap_prior(temp_w,gamma) + disc_adap_prior(temp_n,gamma) + disc_adap_prior(temp_e,gamma) + disc_adap_prior(temp_s,gamma);    
    end
    
end
%% 
% MRF prior: Quadratic function: g_1(u) = |u|^2

function y = quadratic(u)
    y = u.*u;
end

function y = der_quadratic(u)
    y = 2*u;
end
%% 
% MRF prior: Discontinuity-adaptive Huber function:

function y = huber(u,gamma)
    x = abs(u);
    limit = (x <= gamma);
    x(limit) = 0.5 * x(limit).^2;
    x(~limit) = gamma * x(~limit) - 0.5 * gamma.^2; 
    y = x;
end

function y = der_huber(u,gamma)
    x = abs(u);
    limit = (x > gamma);
    x(limit) = gamma * u(limit) ./ x(limit);
    
    y = x;
end
%% 
% MRF prior: Discontinuity-adaptive function:

function y = disc_adap_prior(u,gamma)
    x = abs(u);
    y = gamma*x - log(1+x/gamma)*gamma^2;
end

function y = der_disc_adap_prior(u,gamma)
   y = (gamma*u)./ (gamma + abs(u));
end
%% 
% RRMSE:

function y = RRMSE(A,B)

   N = (abs(A)-abs(B)).^2;
   D = abs(A).^2;
   
   numerator = sqrt(sum(N(:)));
   denominator = sqrt(sum(D(:)));
   
   y = numerator/denominator;
   
end
%% 
% Rician Function:

function y = rician(X,Y,alpha)
    y = 0.5*alpha*(Y.*Y + X.*X) - log(besselj(0,alpha*X.*Y));
end
%% 
% Complex-Gaussian Function:

function y = gaussian(X,Y,alpha)
    y = alpha*abs(Y-X).^2;
end