%%% use NARMAX to predict wind speed, i.e. to identify the non-linear model

clear all;
clc;

%% import data
m = readtable('sample.csv','ReadVariableNames',true,'Delimiter',',');
preset = datenum(m.TIMESTAMP(1),'yyyy/mm/dd HH:MM');
m.time = datenum(m.TIMESTAMP,'yyyy/mm/dd HH:MM') - preset;

%% training set
y = m.WS_10m_Avg(1:2000);
% y = m.WS_10m_Avg;     % data is too large to process...
% y = normalize(m.WS_10m_Avg);

N = size(y,1);
u = zeros(N,1);
e_guess = normrnd(0,std(y),[N,1]);
% e_guess = normrnd(0,std(y)/sqrt(N),[N,1]);    %wait, I though standard
% error...???

%% NARMAX
ny = 2;
nu = 1;
ne = 2;
nl = 2;
narmax = NARMAX(ny, nu, ne, nl);

P = regressor_matrix(narmax, ny, nu, ne, y, u, e_guess);

[W,A] = qr(P);
g = inv(W*W')*W'*y(max(nu,ny)+1:size(y,1));
narmax.process_parameters = linsolve(A,g);

%print of parameters and respective terms
for i=1:size(narmax.full_model,1)
   fprintf('%.4f\t\t\t\t[', narmax.process_parameters(i));
   for j=1:size(narmax.full_model,2)
       fprintf('%d ', narmax.full_model(i,j));
   end
   fprintf(']\n');
end

%% Generate predicted value from the model

%validation set
y = m.WS_10m_Avg(2001:4000);

e_osa = zeros(max(max(nu,ny),ne),1);
Y_OSA(1:max(max(nu,ny),ne)) = y(1:max(max(nu,ny),ne));
for i=max(max(ny,nu),ne):size(y,1)-1
    v_aux = [flip(y(i-ny+1:i)); flip(u(i-nu+1:i)); flip(e_osa(i-ne+1:i))];
    % v_aux = [flip(Y_OSA(i-ny+1:i)'); flip(u(i-nu+1:i)); flip(e_osa(i-ne+1:i))];
    %the above line is totally wrong, NARMAX is an "one-step ahead"
    %prediction, not a "free-run" prediction. Meaning that you can only
    %predict the next time t based on the "real" past t-1, t-2, etc.,
    %instead of the "predicted" past t-1, t-2, etc. predicted from t-2,
    %t-3, etc. Hence the term "one-step ahead", meaning that you can only
    %predict one-step ahead into the future, but no more, until you obtain
    %the real-world value of the next step.
    for j = 1:size(narmax.full_model,1)
        aux = 1;
        for k=1:size(narmax.full_model,2)
            aux = aux*((v_aux(k))^narmax.full_model(j,k));
        end
        R(j) = aux;
    end
    size(R)
    size(narmax.parameters)
    Y_OSA(i+1) = R*narmax.process_parameters;
    e_osa(i+1) = y(i+1)-Y_OSA(i+1);
end

%% Evaluate the result

%root mean square error
RMSE = sqrt(mean((y-Y_OSA').^2));

%plot
hold on
plot(Y_OSA);
plot(y);
title('NARMAX Simulation');
legend('model output', 'Real output');
hold off