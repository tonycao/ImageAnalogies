function yp = prox_fp( y, p, param )

yp = 1/(1+param.sigma)*(param.sigma*y + p );
