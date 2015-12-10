#ifndef __GDD_GQD_INLINE_CU__
#define __GDD_GQD_INLINE_CU__

#define _GQD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _GQD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996


/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|
__host__ __device__
double quick_two_sum( double a , double b, double &err )
{

	if(b == 0.0) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	err = b - (s - a);

	return s;
}

__host__ __device__
double two_sum( double a, double b, double &err )
{

	if( (a == 0.0) || (b == 0.0) ) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	double bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	
	return s;
}


//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
__host__ __device__
double quick_two_diff( double a, double b, double &err )
{
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s;
        
	/*
	if(fabs((a-b)/a) < GPU_D_EPS) {
                s = 0.0;
                err = 0.0;
                return s;
        }
	*/

	s = a - b;
	err = (a - s) - b;
	return s;
}

//computes fl( a - b ) and err( a - b )
__host__ __device__
double two_diff( double a, double b, double &err )
{
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s = a - b;
	
	/*
	if(fabs((a-b)/a) < GPU_D_EPS) {
		s = 0.0;
		err = 0.0;
		return s;
	}
	*/	

	double bb = s - a;
	err = (a - (s - bb)) - (b + bb);
	return s;
}

// Computes high word and lo word of a 
__host__ __device__
void split(double a, double &hi, double &lo) 
{
	double temp;
	if (a > _GQD_SPLIT_THRESH || a < -_GQD_SPLIT_THRESH)
	{
		a *= 3.7252902984619140625e-09;  // 2^-28
		temp = _GQD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
		hi *= 268435456.0;          // 2^28
		lo *= 268435456.0;          // 2^28
	} else 	{
		temp = _GQD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
	}
}

/* Computes fl(a*b) and err(a*b). */
 __device__
double two_prod(double a, double b, double &err) 
{
	
	double a_hi, a_lo, b_hi, b_lo;
	double p = a * b;
	split(a, a_hi, a_lo);
	split(b, b_hi, b_lo);
	
	//err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 
	err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 

	return p;
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
__host__ __device__
double two_sqr(double a, double &err) 
{
	double hi, lo;
	double q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	return q;
}

/* Computes the nearest integer to d. */
__host__ __device__
double nint(double d) 
{
	if (d == floor(d))
		return d;
	return floor(d + 0.5);
}



#endif /* __GDD_GQD_INLINE_CU__ */
