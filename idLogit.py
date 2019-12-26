#!/usr/bin/env python3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import sys

import numpy as np
import pandas as pd
import cvxpy as cp

from scipy.sparse import coo_matrix
from multiprocessing import Pool

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def setup_data( file ) : 

	df = pd.read_csv( file )

	N = df.shape[0]

	unique_ind_ids = df['Session ID'].unique()
	I = len(unique_ind_ids)
	ind_id_map = {}
	for i in range(0,I) : 
		ind_id_map[ unique_ind_ids[i] ] = i
	    
	unique_alt_ids = df['Left Choice ID'].unique()
	A = len(unique_alt_ids)
	alt_id_map = {}
	for i in range(0,A) : 
		alt_id_map[ unique_alt_ids[i] ] = i

	def recast( x ): 
		return ind_id_map[x['Session ID']] , \
					alt_id_map[x['Left Choice ID']] , \
					alt_id_map[x['Right Choice ID']] , \
					1 if x['Winner ID'] == x['Left Choice ID'] else -1

	def code_indices( x ) : 			return ind_id_map[ x['Session ID'] ]
	def code_left_alternatives( x ) : 	return alt_id_map[ x['Left Choice ID'] ]
	def code_right_alternatives( x ) : 	return alt_id_map[ x['Right Choice ID'] ]
	def code_winners( x ): 				return 1 if x['Winner ID'] == x['Left Choice ID'] else -1

	inds = df.apply( code_indices , axis=1 ).values.flatten()

	LR = df[['Left Choice ID','Right Choice ID']].copy()
	LR['Left Choice ID' ] = LR.apply( code_left_alternatives , axis=1 )
	LR['Right Choice ID'] = LR.apply( code_right_alternatives , axis=1 )
	LR = LR.values

	y  = df.apply( code_winners , axis=1 ).values.flatten()

	return I , A , N , inds , LR , y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def make_idLogit( I , A , N , inds , LR , y ) : 
    
    """
    
    Takes in the following arguments: 
    
    I        scalar     number of individuals
    A        scalar     number of alternatives
    N        scalar     number of observations
    inds    N-vector    individual for each observation
    LR    (N,2)-matrix  left and right alternative index (respectively) for each observation
    y       N-vector    choice, coded as +1 for "left" and -1 for "right"
    
    """

    def objective_fcn( A , YX , p , Lambda ):
        return sum( cp.logistic( YX * p ) ) + Lambda * cp.norm( p[A:] , 1 )

    # Construct the problem

    # variables... A beta's plus I * A "idiosyncratic deviations"
    Nvars = (I+1)*A
    p = cp.Variable( Nvars )
    
    # construct outcome-signed data matrix... 

    # outcome-signed data matrix WITHOUT idiosyncratic deviations
    #rows = np.repeat( np.arange(N) , 2 )               # [1,1,2,2,...,N,N]
    #cols = LR.flatten()                                # choices, [L(1),R(1),L(2),R(2),...,L(N),R(N)]
    #data = np.repeat( y , 2 ) * np.tile( [-1,1] , N ) # [-y(1),y(1) ],-y(2),y(2),...,-y(N),y(N)]
    #YX = coo_matrix( (data,(rows,cols)) , shape=(N,A) , dtype=np.float64 )

    # outcome-signed data matrix with idiosyncratic deviations... 4 entries per observation, 
    # row n should be 
    # 
    #     y(n) * ( - p[L(n)] + p[R(n)] - p[ (i(n)+1)*A + L(n) ] + p[ (i(n)+1)*A + R(n) ] )
    # 
    rows , cols , data = np.zeros( 4*N ) , np.zeros( 4*N ) , np.zeros( 4*N )

    rows[  0:  N] = np.arange(N)
    cols[  0:  N] = LR[:,0]
    data[  0:  N] = - y.copy()

    rows[  N:2*N] = rows[0:N].copy()
    cols[  N:2*N] = LR[:,1]
    data[  N:2*N] = y.copy()

    rows[2*N:3*N] = rows[0:N].copy()
    cols[2*N:3*N] = ( inds + 1 ) * A + LR[:,0]
    data[2*N:3*N] = - y.copy()

    rows[3*N:4*N] = rows[0:N].copy()
    cols[3*N:4*N] = ( inds + 1 ) * A + LR[:,1] 
    data[3*N:4*N] = y.copy()

    YX = coo_matrix( (data,(rows,cols)) , shape=(N,Nvars) , dtype=np.float64 )

    # construct constraint matrix... coefficients sum to zero, each deviation vector sums to zero
    
    rows = np.repeat( np.arange(I+1) , A )             # [1,1,..,1,2,2,...,2,...,I+1,I+1,...,I+1]
    cols = np.arange( Nvars )                          # [1,2,...,Nvars]
    data = np.ones( Nvars )                            # [1,1,...,1]
    S = coo_matrix( (data,(rows,cols)) , shape=(I+1,Nvars) , dtype=np.float64 )

    # construct constraint RHS... all zero RHSs
    b = np.zeros( I+1 )
    
    # CVXPy constraint object
    constraints = [ S * p == b ]
    
    # penalty parameter
    Lambda = cp.Parameter( sign="positive" )
    
    # problem
    prob = cp.Problem( cp.Minimize( objective_fcn(A,YX,p,Lambda) ) , constraints )
    return p , Lambda , prob , YX

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def loglik( I , A , N , YX , p_vals ) : 
	return - np.sum( np.log1p( np.exp( YX * p_vals ) ) ) / N

default_lambdas = np.logspace( -4 , 2 , 50 )[::-1]
def trace_idLogit( I , A , N , inds , LR , y , T=50 , lambdas=default_lambdas ) : 

	p , Lambda , prob , YX = make_idLogit( I , A , N , inds , LR , y )

	betas , delta , loglk , success = np.zeros( (A,T) ) , np.zeros( (I*A,T) ) , np.zeros( T ) , np.ones( T , dtype=np.bool )
	for i in range( T ):
		print( i , lambdas[i] )
		Lambda.value = lambdas[i]
		try : 
			prob.solve( solver="ECOS" , warm_start = True )
			betas[:,i] = p.value[0:A].flatten()
			delta[:,i] = p.value[A: ].flatten()
			loglk[ i ] = loglik( I , A , N , YX , p.value )
		except cp.error.SolverError as e : 
			print( e )
			success[i] = False

	return betas , delta , loglk , success

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def temp() : 

	from multiprocessing.dummy import Pool as ThreadPool 

	def write(i, x):
	    print(i, "---", x)

	a = ["1","2","3"]
	b = ["4","5","6"] 

	pool = ThreadPool(2)
	pool.starmap(write, zip(a,b)) 
	pool.close() 
	pool.join()

def selse() : 

	pool = Pool( processes = 10 )

	I , A , N , inds , LR , y = setup_data( sys.argv[1] )
	
	S = 10
	Bs = np.random.choice( N , (S,N) )

	def bsrun( s ) : 
		if( s is None ) : 
			betas , delta = trace_idLogit( I , A , N , inds , LR , y )
		else : 
			betas , delta = trace_idLogit( I , A , N , inds[Bs[:,s]] , LR[Bs[:,s],:] , y[Bs[:,s]] )

	pool.starmap( bsrun , np.arange(S) )

	pool.close() 
	pool.join()

if __name__ == "__main__" : 

	I , A , N , inds , LR , y = setup_data( sys.argv[1] )

	print( I , A , N )
	print( type(I) , type(A) , type(N) )
	print( inds.shape , inds.dtype )
	print( LR.shape , LR.dtype )
	print( y.shape , y.dtype )
	
	# put in "actual" estimates... write to "actual/..."
	
	betas , delta , loglk , success = trace_idLogit( I , A , N , inds , LR , y )
	
	sys.exit()

	# bootstrapping

	S = 500
	Bs = np.random.choice( N , (S,N) )

	for s in range(0,S) : 
		betas , delta , loglk , success = trace_idLogit( I , A , N , inds[Bs[s,:]] , LR[Bs[s,:],:] , y[Bs[s,:]] )
		betas.tofile( "bootstrap/%s/betas-%s.bin" % (sys.argv[2],s) )
		delta.tofile( "bootstrap/%s/delta-%s.bin" % (sys.argv[2],s) )
		loglk.tofile( "bootstrap/%s/loglk-%s.bin" % (sys.argv[2],s) )
		success.astype( np.int ).tofile( "bootstrap/%s/success-%s.bin" % (sys.argv[2],s) )

