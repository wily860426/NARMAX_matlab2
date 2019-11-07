%%% use EMD to decompose windspeed signal 
%%% (either into 3, or completely decompose it into imf's)
%%% then run NARMAX for each
%%% then combine them

clear all;
clc;

m = readtable('sample.csv','ReadVariableNames',true,'Delimiter',',');
preset = datenum(m.TIMESTAMP(1),'yyyy/mm/dd HH:MM');
m.time = datenum(m.TIMESTAMP,'yyyy/mm/dd HH:MM') - preset;

y = m.WS_10m_Avg(1:2000);

%% decompose original signal into 3 

% addpath('...EMD');
% savepath;

%use EMD to decompose windspeed signal y
imf = emd(y);
% N = size(y,1);
% t = 1:N;
% emd_sum_visu(y,t,imf,1,10);

y1 = imf(1,:) + imf(2,:) + imf(3,:);
y2 = imf(4,:) + imf(5,:) + imf(6,:);
y3 = imf(7,:) + imf(8,:) + imf(9,:) + imf(10,:);

%% implement NARMAX, and evaluate the result using the origial training set
y1_hat = for_project2(y1');
y2_hat = for_project2(y2');
y3_hat = for_project2(y3');

y_hat = y1_hat + y2_hat + y3_hat;

RMSE = sqrt(mean((y-y_hat').^2));

hold on
plot(y_hat);
plot(y);
title('EMD-NARMAX Simulation');
legend('model output', 'Real output');
hold off

%% completely decompose orignal signal into imf's, implement NARMAX, then evaluate
% imf = emd(y);
% y_hat = zeros(size(imf));
% n = size(imf,1);
% for rows = 1:n
%     y_hat(rows,:) = for_project2(imf(rows,:)');
% end
% y_hat2 = sum(y_hat);
% RMSE = sqrt(mean((y-y_hat2').^2));
% 
% hold on
% plot(y_hat2);
% plot(y);
% title('EMD-NARMAX Simulation');
% legend('model output', 'Real output');
% hold off