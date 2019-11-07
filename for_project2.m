%%% given y
%%% implement NARMAX, return y_hat, the predicted y

function [y_hat] = for_project2(y)

N = size(y,1);
u = zeros(N,1);
e_guess = normrnd(0,std(y),[N,1]);

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
for i=max(max(ny,nu),ne):size(y,1)-1
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

y_hat = Y_OSA;

end