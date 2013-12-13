function yp = prox_fv( yv, yw, param )

%yp = 1/(1+options.sigma)*(options.sigma*y + p );

Iv = eye(size(yv,1));
Iw = eye(size(yw,1));
mat = [Iv+param.gamma*Iv -Iw;-Iv Iw+param.gamma*Iw];
yp = param.gamma*inv(mat)*[yv;yw];