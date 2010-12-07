#ifndef __GDD_LOG_CU__
#define __GDD_LOG_CU__

#include "common.cu"

/* Logarithm.  Computes log(x) in double-double precision.
   This is a natural logarithm (i.e., base e).            */
__device__
gdd_real log(const gdd_real &a) {
  
	if (is_one(a)) {	
		return make_dd(0.0);
	}

//!!!!!!!!!
//TO DO: return an errro
	if (a.x <= 0.0) {
		//return _nan;
		return make_dd( 0.0 );
	}

	gdd_real x = make_dd(log(a.x));   // Initial approximation 

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

#endif /* __GDD_LOG_CU__ */


