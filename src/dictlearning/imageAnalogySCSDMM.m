function [IOut, IOutPredict] = imageAnalogySCSDMM(inputImg, matDictionary, param)

%% initialize parameters

fprintf('initializing parameters...');

%vecPatches = extractpatchesvector(rawImage, param);
[NN1, NN2] = size(inputImg);
bb = param.patchSideLength;
slidingDis = 1;
maxBlocksToConsider = param.maxBlocksToConsider;
maxNumBlocksToTrainOn = param.maxNumBlocksToTrainOn;
K = param.dictSize;

while (prod(floor((size(inputImg)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[vecPatches, idx] = my_im2col(inputImg,[bb,bb],slidingDis);
%blocksPredict = zeros(size(blocks)); 

patchSize = param.patchSideLength^2;
%nrPatches(1) = floor(size(rawImage,1)/param.patchSideLength);
%nrPatches(2) = floor(size(rawImage,2)/param.patchSideLength);
numberofPatches = size(idx,2);%nrPatches(1)*nrPatches(2);

f = inputImg(:);
u = zeros(size(vecPatches));
alphas = (rand(param.dictSize, numberofPatches)-0.5)*2;
D = matDictionary(1:patchSize,:);
%R = zeros(patchSize, size(f,1));%, numberofPatches);
R = sparse(patchSize, size(f,1));

%for iI = 1:numberofPatches
%    offset = (iI-1)*patchSize;
%    R(:,offset+1:offset+patchSize, iI) = eye(patchSize);
%end

% setting up the temporary variables

su = sparse( size(f,1), 1);
zu = sparse( size(f,1), 1);
yu = sparse( size(f,1), 1);

sv = sparse( patchSize,numberofPatches);
zv = sparse( patchSize,numberofPatches);
yv = sparse( patchSize,numberofPatches);

sw = sparse( patchSize, numberofPatches );
zw = sparse( patchSize, numberofPatches );
yw = sparse( patchSize, numberofPatches );

sq = sparse( param.dictSize, numberofPatches );
zq = sparse( param.dictSize, numberofPatches );
yq = sparse( param.dictSize, numberofPatches );

%% projection matrices
%R = eye(patchSize);
%I = eye(patchSize);
%D = matDictionary;
%Q = inv( I + sigma(R'*R))

tic
% Q = zeros(size(R,2), size(R,2));
Q = sparse(size(R,2), size(R,2));
idxMat = zeros(size(inputImg)-[bb, bb]+1);
%idxMat([[1:slidingDis:end-1],end],[[1:slidingDis:end-1],end]) = 1; % take blocks in distances of 'slidingDix', but always take the first and last one (in each row and column).
%idx = find(idxMat);
[rows,cols] = ind2sub(size(idxMat),idx);
%matlabpool 
for iI = 1:numberofPatches
    I = sparse(size(inputImg,1), size(inputImg,2));
    I(rows(iI):rows(iI)+bb-1,cols(iI):cols(iI)+bb-1) = 1;
    RTR = I(:);
    Q = Q + sparse(diag(I(:)));
end
Q = inv(speye(size(Q,1))+Q);
%Q = 1./(speye(size(Q,1))+Q);
invDTDI = inv(sparse(D'*D)+speye(param.dictSize));
toc
%matlabpool close
currentEnergy = 10000;

notConverged = true;
numberofIterations = 0;

fprintf('start sdmm iterations...');

for i = 1:10
%while notConverged
    tic
    %% average
    % for U
    sigmaR = sparse(size(f,1), 1);
    for iI = 1:numberofPatches
        I = sparse(size(I,1), size(I,2));
        I(rows(iI):rows(iI)+bb-1,cols(iI):cols(iI)+bb-1) = 1;
        iInd = 1:bb^2;
        jInd = find(I==1);
        a = ones(bb^2,1);
        R = sparse(iInd',jInd, a, patchSize, size(f,1)); %I(:);
        sigmaR = sigmaR + R'*(yv(:,iI)-zv(:,iI));
    end
    U = Q*(yu-zu)+Q*sigmaR;
    
    % for ai
    for iI = 1:numberofPatches
        alphas(:,iI) = invDTDI*(D'*(yw(:,iI)-zw(:,iI))+(yq(:,iI)-yq(:,iI)));
    end
    
    %% update
    % for U
    su = U;
    yu = prox_fp(su+zu, f, param);
    zu = zu + su -yu;
    
    tempEnergy = 0;
    
    for iI = 1:numberofPatches
        % for vi and wi
        I = sparse(size(I,1), size(I,2));
        I(rows(iI):rows(iI)+bb-1,cols(iI):cols(iI)+bb-1) = 1;
        iInd = 1:bb^2;
        jInd = find(I==1);
        a = ones(bb^2,1);
        R = sparse(iInd',jInd, a, patchSize, size(f,1));
        sv(:,iI) = R*U;
        sw(:,iI) = D*alphas(:,iI);
        
        yvw = prox_fv(sv(:,iI)+zv(:,iI), sw(:,iI)+zw(:,iI), param);
        
        yv(:,iI) = yvw(1:patchSize);
        yw(:,iI) = yvw(patchSize+1:end);
        
        zv(:,iI) = zv(:,iI) + sv(:,iI) - yv(:,iI);
        zw(:,iI) = zw(:,iI) + sw(:,iI) - yw(:,iI);
        
        % for q
        sq(:,iI) = alphas(:,iI);
        yq(:,iI) = prox_fs(sq(:,iI)+zq(:,iI), param);
        zq(:,iI) = zq(:,iI) + sq(:,iI) - yq(:,iI);
        
        tempEnergy = tempEnergy + 0.5*norm((R*U-D*alphas(:,iI)),2)^2;
        
    end
    
    % Evaluate energy
    
    lastEnergy = currentEnergy;
    
    currentEnergy = 0.5*norm((U-f),2)^2+(tempEnergy + param.lambda*norm( alphas(:) ,1 ))/numberofPatches;
    
    fprintf( 'Iter %d : E = %f\n', numberofIterations+1, currentEnergy );
    
    absoluteEnergyReduction = lastEnergy - currentEnergy;
    relativeEnergyReduction = absoluteEnergyReduction / currentEnergy;
    
    energyDecreased = true;
    if absoluteEnergyReduction<0
        energyDecreased = false;
    end
    
    if numberofIterations == 0
        initialEnergy = currentEnergy;
    end
    
    if energyDecreased &&(absoluteEnergyReduction <= param.absoluteEnergyStoppingTolerance || ...
            relativeEnergyReduction <= param.relativeEnergyStoppingTolerance)
        
        %fprintf('Energy decrease below tolerance. Done iterating.\n\n');
        break;
    end
    
    numberofIterations = numberofIterations + 1;
    
    if numberofIterations >= param.maxNumberofIterations
        notConverged = false;
    end
    toc
end

U = reshape(U, [NN1, NN2]);

WeightPredict = zeros(NN1,NN2);
IMoutPredict = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(inputImg)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);        
    alphaInd = (cols(i)-1)*max(cols(:))+rows(i)
    block = reshape( matDictionary(1+bb^2:end,:)*alphas(:,alphaInd), bb, bb);
    IMoutPredict(row:row+bb-1,col:col+bb-1)=IMoutPredict(row:row+bb-1,col:col+bb-1)...
        +block;
    WeightPredict(row:row+bb-1,col:col+bb-1)=WeightPredict(row:row+bb-1,col:col+bb-1)...
        +ones(bb);
    %count = count+1;
end;

IOut = U;
IOutPredict = IMoutPredict./WeightPredict;

