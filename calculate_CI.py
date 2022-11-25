from scipy.stats import beta
from scipy.optimize import fsolve

def full_function(alpha,N,pvalue,qvalue):
    return(beta.ppf(pvalue,alpha*N+1,(1-alpha)*N+1)-qvalue)

for N in [10000]:
    for pvalue in [.025,.975]:
        print(fsolve(full_function,.5,args=(N,pvalue,.05))[0])