%% synthetic noise

function noisePatches = generateNoisePatches2(nPatches, psl, pintensity, pstd)

%nmeans1 = random('norm', intensity(1), variance(1));
%nmeans2 = random('norm', intensity(2), variance(2));


numPatches = size(nPatches,2);

ind      = randperm(numPatches, fix(numPatches/2));

noiseMean = bsxfun(@plus, pintensity, 0.1*randn(2,numPatches));

noisePatch1 = bsxfun(@plus, noiseMean(1,:), pstd(1)*randn(psl^2, numPatches));
noisePatch2 = bsxfun(@plus, noiseMean(2,:), pstd(2)*randn(psl^2, numPatches));


noisePatches = [noisePatch1; noisePatch2];

noisePatches(1:psl^2,ind) = nPatches(1:psl^2,ind); 