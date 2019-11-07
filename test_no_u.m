clear all;
clc;

N = 300;
e = 0.1*randn(N,1);
y = zeros(N,1);

y(2) = -0.605*y(1) + e(2);
for k = 3:N
    y(k) = -0.605*y(k-1) - 0.163*y(k-2)^2 - 0.4*e(k-1) + e(k);
end

e_guess = normrnd(0,std(y),[N,1]);
% e_guess = normrnd(0,std(y)/sqrt(N),[N,1]);    %wait, I though standard
% error...???
u = zeros(N,1); %or ones?

ny = 2;
nu = 1;
ne = 2;
nl = 2;
narmax = NARMAX(ny, nu, ne, nl);

P = regressor_matrix(narmax, ny, nu, ne, y, u, e_guess);

[W,A] = qr(P);
g = inv(W*W')*W'*y(max(nu,ny)+1:size(y,1));
narmax.process_parameters = linsolve(A,g);

e_osa = zeros(max(max(nu,ny),ne),1);
Y_OSA(1:max(max(nu,ny),ne)) = y(1:max(max(nu,ny),ne));
for i=max(max(ny,nu),ne):size(y,1)-1 % For each moving horizon window on y/u
    v_aux = [flip(y(i-ny+1:i)); flip(u(i-nu+1:i)); flip(e_osa(i-ne+1:i))];
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

hold on
plot(Y_OSA)
plot(y)
title('Simulação NARMAX');
legend('model output', 'Real output');
hold off
