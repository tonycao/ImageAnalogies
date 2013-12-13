function [alphas] = sparsecodingSDMM(vecPatches, matDictionary, param)

sparsity = zeros(param.maxNumPatchesToTrainOn,1);

%matlabpool 
for i = 1:param.maxNumPatchesToTrainOn
    
    %progressbar(i,param.maxNumPatchesToTrainOn);
    
    %iterstr = strcat(['Iter approx = ', num2str(i/param.maxNumPatchesToTrainOn)]);
    
    %fprintf(iterstr);
    
    alphas(:,i) = sparsecodingpatchesSDMM(vecPatches(:,i), matDictionary, param);
    
    sparsity(i) = sum(abs(alphas(:,i))>1e-3)/param.dictSize;
    
    %fprintf( repmat('\b', [1 length(iterstr)]) );
    %fprintf('Iter approx = %d\n\n', i/param.maxNumPatchesToTrainOn);
        % record energies
    %fprintf('\n');
    if 1
        progressbar(i,param.maxNumPatchesToTrainOn);
    end
    %fprintf('\n');
end

%matlabpool close
% compute overall sparsity
overallSparsity = sum(sparsity)/param.maxNumPatchesToTrainOn;

fprintf('\nAverage sparsity level (0: sparse; 1:dense) = %4.2d\n', overallSparsity);


function [alphas] = sparsecodingpatchesSDMM(vecPatch, matDictionary, param)
                    
%% sparse coding done with SDMM

% setting up the temporary variables

sp = zeros( param.patchSideLength^2*param.numberofImages, 1 );
zp = zeros( param.patchSideLength^2*param.numberofImages, 1 );
yp = zeros( param.patchSideLength^2*param.numberofImages, 1 );

sq = zeros( param.dictSize, 1 );
zq = zeros( param.dictSize, 1 );
yq = zeros( param.dictSize, 1 );

D = matDictionary;
% projection matrix
P = D'*D + eye(param.dictSize);

currentEnergy = 10000;
%lastEnergy = 10000;
%initialEnergy = 10000;

notConverged = true;
numberofIterations = 0;

while notConverged
    % averaging 
    % alphas = inv(P)*( D'*(yp-zp) + (yq-zq) );
    alphas = P\( D'*(yp-zp) + (yq-zq) );
    % for p
    % create intermediate transformed variable
    sp = D*alphas;
    
    % update it
    yp = prox_fp( sp + zp, vecPatch, param );
    
    % update the dual variables
    zp = zp + sp - yp;
    
    % for q
    % create intermediate transformed variable
    sq = alphas;
    
    % update it
    yq = prox_fs( sq + zq, param );
    
    % update the dual variables
    zq = zq + sq - yq;
    
    % compute current energy
    % E = 1/2\| v-D\alpha\|^2 + \lambda \|\alpha\|_1 = 1/2\| v-sp\|^2 + \lambda\|sq\|_1
    
    lastEnergy = currentEnergy;
    
    currentEnergy = 0.5*norm( vecPatch-D*alphas, 2)^2 + param.lambda*norm( alphas ,1 );
    
    %fprintf( 'Iter %d : E = %f\n', numberofIterations+1, currentEnergy );
    
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
    
    % test convergence 
    if numberofIterations >= param.maxNumberofIterations
        notConverged = false;
    end
    
end
