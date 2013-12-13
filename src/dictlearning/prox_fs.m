function yq = prox_fs( y, param )

yq = zeros( size( y ) );

indx = find( y>param.lambda/param.sigma );
yq( indx ) = y( indx ) - param.lambda/param.sigma;

indx = find( y<-param.lambda/param.sigma );
yq( indx ) = y( indx ) + param.lambda/param.sigma;
