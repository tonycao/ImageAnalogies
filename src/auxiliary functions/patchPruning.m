function newPatch = patchPruning(oldPatch, oldPatch2, threshold)

pvars    = var(oldPatch, 0, 1);

idx      = pvars > threshold;

newPatch = oldPatch2(:, idx);