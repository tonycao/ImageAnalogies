%% synthetic noise

function noisePatches = generateNoisePatches(nPatches, psl, pintensity, pstd)

%nmeans1 = random('norm', intensity(1), variance(1));
%nmeans2 = random('norm', intensity(2), variance(2));

noiseMean = bsxfun(@plus, pintensity, 0.1*randn(2,nPatches));

noisePatch1 = bsxfun(@plus, noiseMean(1,:), pstd(1)*randn(psl^2, nPatches));
noisePatch2 = bsxfun(@plus, noiseMean(2,:), pstd(2)*randn(psl^2, nPatches));

noisePatches = [noisePatch1; noisePatch2];