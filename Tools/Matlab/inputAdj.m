function [adj, nodeNum, edgeNum] = inputAdj(adjFile)

% open file
[fid,msg] = fopen(adjFile,'r');
if fid == -1
    disp(msg);
    return;
end

% read N and M
A = fscanf(fid,'%d',2);
nodeNum = A(1);
edgeNum = A(2);

% read all edges
A = fscanf(fid,'%d%d',[2,inf]);
A = A';

% store edges into adj
adj=sparse([]);
for i = 1:edgeNum,
    if A(i,1) ~= A(i,2) % not self cycle
        adj(A(i,1),A(i,2)) = 1;
        adj(A(i,2),A(i,1)) = 1;
    end
end

fclose(fid);

clear adjFile fid msg A i;
