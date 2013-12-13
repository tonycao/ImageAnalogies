clear all;

I=double(imread('data/lena.png'))/255;
% extract 8 x 8 patches
X=im2col(I,[10 10],'sliding');
X=X-repmat(mean(X),[size(X,1) 1]);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

param.K=500;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=4; % number of threads
param.batchsize=1000;

param.iter=1000;  % let us see what happens after 1000 iterations.

%%%%%%%%%% FIRST EXPERIMENT %%%%%%%%%%%
tic
D = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

param.approx=0;
fprintf('Evaluating cost function...\n');
tic
alpha  = mexLasso(X,D,param);
toc
tic
alpha2 = mexCD(X,D,sparse(size(alpha,1),size(alpha,2)),param);
toc
% initialize parameter for our method
tic
param.patchSideLength = 10;
param.RR              = 5;
param.dictSize        = param.RR*param.patchSideLength^2;
param.sigma           = 1;
param.rawpatchDict    = 0;
param.mu              = 1;
param.numberofImages  = 2;
param.slidingDis      = 2;
%param.normalized      = 0;
param.reduceMean      = 0;
param.saveIntermediateDictionaryResults = false;
param.relativeEnergyStoppingTolerance   = 1e-4;
param.absoluteEnergyStoppingTolerance   = 1e-2;
param.maxNumberofIterations             = 1000;
param.maxNumberofDictionaryUpdate       = 1;
param.maxNumBlocksToTrainOn             = 100000;
param.maxBlocksToConsider               = 250000;
param.lasso = 1;
%[~, rectOutput] = patchReconstructSC(X, D, param);
toc
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
R2=mean(0.5*sum((X-D*alpha2).^2)+param.lambda*sum(abs(alpha2)));
%R3=mean(0.5*sum((X-D*rectOutput.alphas).^2)+param.lambda*sum(abs(rectOutput.alphas)));
ImD=displayPatches(D);
subplot(1,3,1);
imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);
fprintf('objective function: %f\n',R2);
%fprintf('objective function: %f\n',R3);
drawnow;

fprintf('*********** SECOND EXPERIMENT ***********\n');
%%%%%%%%%% SECOND EXPERIMENT %%%%%%%%%%%
% Train on half of the training set, then retrain on the second part
X1=X(:,1:floor(size(X,2)/2));
X2=X(:,floor(size(X,2)/2):end);
param.iter=500;
tic
[D model] = mexTrainDL(X1,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);
tic
% Then reuse the learned model to retrain a few iterations more.
param2=param;
param2.D=D;
[D model] = mexTrainDL(X2,param2,model);
%[D] = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);

% let us add sparsity to the dictionary itself
fprintf('*********** THIRD EXPERIMENT ***********\n');
param.modeParam=0;
param.iter=1000;
param.gamma1=0.3;
param.modeD=1;
tic
[D] = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);
tic
subplot(1,3,2);
ImD=displayPatches(D);
imagesc(ImD); colormap('gray');
drawnow;

fprintf('*********** FOURTH EXPERIMENT ***********\n');
param.modeParam=0;
param.iter=1000;
param.gamma1=0.3;
param.modeD=3;
tic
[D] = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);
tic
subplot(1,3,3);
ImD=displayPatches(D);
imagesc(ImD); colormap('gray');
drawnow;
