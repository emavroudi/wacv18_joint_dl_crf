function y = normcols(x)
%NORMCOLS Normalize matrix columns.
%  Y = NORMCOLS(X) normalizes the columns of X to unit length, returning
%  the result as Y.
%
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  April 2009


y = x*diag(1./sqrt(sum(x.*x)));
