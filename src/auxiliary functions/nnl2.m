function [ind] = nnl2(inputVec, vecSet)
% nearest neighbor search based on L2 norm

n2 = dist2(inputVec, vecSet);

[ns, ind] = sort(n2, 1);

ind = ind(1,:);

end