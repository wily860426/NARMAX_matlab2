%%% given the training (first 2000) + validation set (last 2000) y,
%%% implement NARMAX, return y_val_hat, the predicted y_val

function [y_val_hat] = for_project3(y)

y_train = y(1:2000);
y_val = y(2001:4000);

N = size(y_train,1);
u = zeros(N,1);
e_guess = normrnd(0,std(y_train),[N,1]);

ny = 2;
nu = 1;
ne = 2;
nl = 2;
narmax = NARMAX(ny, nu, ne, nl);

P = regressor_matrix(narmax, ny, nu, ne, y_train, u, e_guess);

[W,A] = qr(P);
g = inv(W*W')*W'*y_train(max(nu,ny)+1:N);
narmax.process_parameters = linsolve(A,g);

e_osa = zeros(max(max(nu,ny),ne),1);
Y_OSA(1: max(max(nu,ny),ne) ) = y_val(1:max(max(nu,ny),ne));
for i=max(max(ny,nu),ne):size(y_val,1)-1
    v_aux = [flip(y_val(i-ny+1:i)); flip(u(i-nu+1:i)); flip(e_osa(i-ne+1:i))];
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
    e_osa(i+1) = y_val(i+1)-Y_OSA(i+1);
end

y_val_hat = Y_OSA;

end