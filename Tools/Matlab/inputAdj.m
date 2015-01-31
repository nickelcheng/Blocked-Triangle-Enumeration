function adj = inputAdj2(adjFile,N)

idx = load(adjFile, '-ascii');
i = idx(:,1);
j = idx(:,2);
a = [i;j];
b = [j;i];
len =length(idx) * 2;
val(1:len) = 1;
adj = sparse(a,b,val,N,N);
