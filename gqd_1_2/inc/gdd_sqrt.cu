#ifndef __GDD_SQRT_CU__
#define __GDD_SQRT_CU__

#include "common.cu"


/* Computes the square root of the double-double number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gdd_real sqrt(const gdd_real &a)
{
	if (is_zero(a))
    		return make_dd(0.0);

  	//TODO: should make an error
  	if (is_negative(a)) {
    		//return _nan;
         	 return make_dd( 0.0 );
  	}

  	double x = 1.0 / sqrt(a.x);
  	double ax = a.x * x;

  	return dd_add(ax, (a - sqr(ax)).x * (x * 0.5));
  	//return a - sqr(ax);
}

#endif /* __GDD_SQRT_CU__ */


