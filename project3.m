%%% project 2, but use validation data to evaluate the model

%{
why use validation data? The trained model is actually trained to best describe 
the structure of the training set, instead of any independent data set. 
therefore, with validation set, we can avoid overfitting, generalize the
model to any independent data set, and assess its predictive power.
%}

clear all;
clc;

m = readtable('sample.csv','ReadVariableNames',true,'Delimiter',',');
preset = datenum(m.TIMESTAMP(1),'yyyy/mm/dd HH:MM');
m.time = datenum(m.TIMESTAMP,'yyyy/mm/dd HH:MM') - preset;

%training + validation data
y = m.WS_10m_Avg(1:4000);

%% decompose original signal into 3 

% %use EMD to decompose windspeed signal y
% imf = emd(y);
% % N = size(y,1);
% % t = 1:N;
% % emd_sum_visu(y,t,imf,1,12);
% 
% y1 = imf(3,:) + imf(4,:);   %discard those high frequency noise
% y2 = imf(5,:) + imf(6,:) + imf(7,:) + imf(8,:);
% y3 = imf(9,:) + imf(10,:) + imf(11,:) + imf(12,:);

%% train NARMAX with training set, and evaluate the result using the validation set
% y1_val_hat = for_project3(y1');
% y2_val_hat = for_project3(y2');
% y3_val_hat = for_project3(y3');
% 
% y_val_hat = y1_val_hat + y2_val_hat + y3_val_hat;
% 
% %validation data
% y_val = y(2001:4000);
% RMSE = sqrt(mean((y_val-y_val_hat').^2));
% 
% hold on
% plot(y_val_hat);
% plot(y_val);
% title('EMD-NARMAX Simulation');
% legend('model output', 'Real output');
% hold off

%% completely decompose y into all imf's, train NARMAX, then evaluate with validation set 
imf = emd(y);
y_val_hat = zeros(size(imf,1),size(imf,2)/2);
n = size(imf,1);
for rows = 2:n      %discard the noise with highest frequency, i.e. the first imf
    y_val_hat(rows,:) = for_project3(imf(rows,:)');
end
y_val_hat2 = sum(y_val_hat);

%because windspeed cannot be negative, set those negative windspeed to zero
for i = 1:size(y_val_hat2,2)
    if y_val_hat2(i) < 0 
        y_val_hat2(i) = 0;
    end
end

%validation data
y_val = y(2001:4000);

%evaluate
RMSE = sqrt(mean((y_val-y_val_hat2').^2));

hold on
plot(y_val_hat2);
plot(y_val);
title('EMD-NARMAX Simulation');
legend('model output', 'Real output');
hold off
