load('../data/assignmentImageDenoisingPhantom.mat');
initial_noise = rrmse(imageNoiseless, imageNoisy);

%Calculations for Quadratic model
alpha_quad = 0.055;
[est_quad, series_quad] = GradientDescent(imageNoisy, 1, alpha_quad, 0);
[est2] = GradientDescent(imageNoisy, 1, 1.2 * alpha_quad, 0);
[est3] = GradientDescent(imageNoisy, 1, 0.8 * alpha_quad, 0);
est_noise_quad = rrmse(imageNoiseless, est_quad);
est2_noise_quad = rrmse(imageNoiseless, est2);
est3_noise_quad = rrmse(imageNoiseless, est3);

%Calculations for Discontinuity-adaptive Huber Function
alpha_huber = 0.99;
gamma_huber = 0.001;
[est_huber, series_huber] = GradientDescent(imageNoisy, 2, alpha_huber, gamma_huber);
[est2] = GradientDescent(imageNoisy, 2, 1.2 * alpha_huber, gamma_huber);
[est3] = GradientDescent(imageNoisy, 2, 0.8 * alpha_huber, gamma_huber);
[est4] = GradientDescent(imageNoisy, 2, alpha_huber, 1.2 * gamma_huber);
[est5] = GradientDescent(imageNoisy, 2, alpha_huber, 0.8 * gamma_huber);
est_noise_huber = rrmse(imageNoiseless, est_huber);
est2_noise_huber = rrmse(imageNoiseless, est2);
est3_noise_huber = rrmse(imageNoiseless, est3);
est4_noise_huber = rrmse(imageNoiseless, est4);
est5_noise_huber = rrmse(imageNoiseless, est5);

%Calculations for Discontinuity-adaptive function
alpha_adap = 0.993;
gamma_adap = 0.0013;
[est_adap, series_adap] = GradientDescent(imageNoisy, 3, alpha_adap, gamma_adap);
[est2] = GradientDescent(imageNoisy, 3, 1.2 * alpha_adap, gamma_adap);
[est3] = GradientDescent(imageNoisy, 3, 0.8 * alpha_adap, gamma_adap);
[est4] = GradientDescent(imageNoisy, 3, alpha_adap, 1.2 * gamma_adap);
[est5] = GradientDescent(imageNoisy, 3, alpha_adap, 0.8 * gamma_adap);
est_noise_adap = rrmse(imageNoiseless, est_adap);
est2_noise_adap = rrmse(imageNoiseless, est2);
est3_noise_adap = rrmse(imageNoiseless, est3);
est4_noise_adap = rrmse(imageNoiseless, est4);
est5_noise_adap = rrmse(imageNoiseless, est5);

%Plotting the functions using jet colormap

figure; h = surf(imageNoiseless);
set(h,'LineStyle','None');
title('Noiseless Image');
colormap('jet');
figure; h = surf(imageNoisy);
set(h,'LineStyle','None');
title('Noisy Image');
colormap('jet');
figure; h = surf(est_quad);
set(h,'LineStyle','None');
title('Quadratic Prior Denoising');
colormap('jet');
figure; h = surf(est_huber);
set(h,'LineStyle','None');
title('Huber Prior Denoising');
colormap('jet');
figure; h = surf(est_adap);
set(h,'LineStyle','None');
title('Discontinuity-Adaptive Prior Denoising');
colormap('jet');

%Plotting the decrease in objective function as Gradient descent progresses

figure; plot(series_quad);
title('Gradient Descent with Quadratic Prior');
figure; plot(series_huber);
title('Gradient Descent with Huber Prior');
figure; plot(series_adap);
title('Gradient Descent with Discontinuity-Adaptive Prior');

function [est, series] = GradientDescent(imageNoisy, type, alpha, gamma)
    step_size = 0.01;
    est = imageNoisy;
    series = zeros(1,100);
    for i = 1:100
        [val, der] = penalty(est, imageNoisy, type, gamma, alpha);
        series(1,i) = sum(val,'all');
        new_est = est - der * step_size;
        [new_val] = penalty(new_est, imageNoisy, type, gamma, alpha);
        if(sum(new_val,'all') < series(1,i))
            est = new_est;
            step_size = 1.1 * step_size;
        else
            step_size = step_size / 2;
        end
    end
end

function [val,der] = penalty(x,y,type, gamma, alpha)
    val = (1 - alpha) * logGaussian(x,y) + alpha * regularLog(x,type, gamma);
    der = (1 - alpha) * derivlogGaussian(x,y) + alpha * derivregularLog(x,type,gamma);
end

function [val] = logGaussian(x,y)
    %val = (y.^2 + x.^2)/2 - log(besselj(0,y.*x));
    val = (y - x).^2 / 2;
end

function [val] = derivlogGaussian(x,y)
    %val = x - y.*(besselj(1,y.*x)./besselj(0,y.*x));
    val = (x - y);
end

function [val] = regularLog(x,type,gamma)
    if(type == 1)
        val = (x - circshift(x,1,2)).^2 + (x - circshift(x,1,1)).^2 + (x - circshift(x,-1,1)).^2 + (x - circshift(x,-1,2)).^2;
    end
    if(type == 2)
        diff1 = abs(x - circshift(x,1,1));
        diff2 = abs(x - circshift(x,1,2));
        diff3 = abs(x - circshift(x,-1,1));
        diff4 = abs(x - circshift(x,-1,2));
        val1 = zeros(size(x,1),size(x,2));
        val2 = zeros(size(x,1),size(x,2));
        val3 = zeros(size(x,1),size(x,2));
        val4 = zeros(size(x,1),size(x,2));
        case1 = diff1 <= gamma;
        case2 = diff2 <= gamma;
        case3 = diff3 <= gamma;
        case4 = diff4 <= gamma;
        val1(~case1) = gamma * diff1(~case1) - 0.5 * (gamma^2);
        val1(case1) = 0.5 * diff1(case1).^2;
        val2(~case2) = gamma * diff2(~case2) - 0.5 * (gamma^2);
        val2(case2) = 0.5 * diff2(case2).^2;
        val3(~case3) = gamma * diff3(~case3) - 0.5 * (gamma^2);
        val3(case3) = 0.5 * diff3(case3).^2;
        val4(~case4) = gamma * diff4(~case4) - 0.5 * (gamma^2);
        val4(case4) = 0.5 * diff4(case4).^2;
        val = val1 + val2 + val3 + val4;
    end
    if(type == 3)
        diff1 = abs(x - circshift(x,1,1));
        diff2 = abs(x - circshift(x,1,2));
        diff3 = abs(x - circshift(x,-1,1));
        diff4 = abs(x - circshift(x,-1,2));
        val = gamma * diff1 - gamma^2 * log(1 + diff1/gamma) + gamma * diff2 - gamma^2 * log(1 + diff2/gamma) + gamma * diff3 - gamma^2 * log(1 + diff3/gamma) + gamma * diff4 - gamma^2 * log(1 + diff4/gamma);
    end
end

function [val] = derivregularLog(x,type,gamma)
    if(type == 1)
        val = 2 * (x - circshift(x,1,2)) + 2 * (x - circshift(x,-1,2)) + 2 * (x - circshift(x,-1,1)) +2 * (x - circshift(x,1,1)); 
    end
    if(type == 2)
        diff1 = abs(x - circshift(x,+1,1));
        diff2 = abs(x - circshift(x,+1,2));
        diff3 = abs(x - circshift(x,-1,1));
        diff4 = abs(x - circshift(x,-1,2));
        val1 = zeros(size(x,1),size(x,2));
        val2 = zeros(size(x,1),size(x,2));
        val3 = zeros(size(x,1),size(x,2));
        val4 = zeros(size(x,1),size(x,2));
        case1 = diff1 <= gamma;
        case2 = diff2 <= gamma;
        case3 = diff3 <= gamma;
        case4 = diff4 <= gamma;
        val1(case1) = diff1(diff1 <= gamma);
        val1(~case1) = gamma;
        val1 = val1 .* sign(x - circshift(x,+1,1));
        val2(case2) = diff2(diff2 <= gamma);
        val2(~case2) = gamma;
        val2 = val2 .* sign(x - circshift(x,+1,2));
        val3(case3) = diff3(diff3 <= gamma);
        val3(~case3) = gamma;
        val3 = val3 .* sign(x - circshift(x,-1,1));
        val4(case4) = diff4(diff4 <= gamma);
        val4(~case4) = gamma;
        val4 = val4 .* sign(x - circshift(x,-1,2));
        val = val1 + val2 + val3 + val4;
    end
    if(type == 3)
        diff1 = abs(x - circshift(x,1,1));
        diff2 = abs(x - circshift(x,1,2));
        diff3 = abs(x - circshift(x,-1,1));
        diff4 = abs(x - circshift(x,-1,2));
        val1 = sign(x - circshift(x,1,1)) .* (gamma + gamma*ones(size(x,1),size(x,2))./(1 + diff1 / gamma));
        val2 = sign(x - circshift(x,1,2)) .* (gamma + gamma*ones(size(x,1),size(x,2))./(1 + diff2 / gamma));
        val3 = sign(x - circshift(x,-1,1)) .* (gamma + gamma*ones(size(x,1),size(x,2))./(1 + diff3 / gamma));
        val4 = sign(x - circshift(x,-1,2)) .* (gamma + gamma*ones(size(x,1),size(x,2))./(1 + diff4 / gamma));
        val = val1 + val2 + val3 + val4;
    end
end

function [err] = rrmse(a, b)
    diff = (abs(a) - abs(b)).^2;
    val = a.^2;
    err = sqrt(sum(diff,'all'))/sqrt(sum(val,'all'));
end