#ifndef __GDD_BASIC_CU__
#define __GDD_BASIC_CU__

/**
 * arithmetic operators
 * comparison
 */

#include "common.cu"

///////////////////// Addition /////////////////////

__device__
gdd_real negative( const gdd_real &a )
{
	return make_dd( -a.x, -a.y );
}

/* double-double = double + double */
__device__
gdd_real dd_add(double a, double b) 
{
	double s, e;
	s = two_sum(a, b, e);
	return make_dd(s, e);
}

/* double-double + double */
__host__ __device__
gdd_real operator+(const gdd_real &a, double b) 
{
	double s1, s2;
	s1 = two_sum(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_dd(s1, s2);
}

/* double + double-double */
__host__ __device__
gdd_real operator+(const double &a, gdd_real b) 
{
	return b + a;
}

__inline__ __host__ __device__
gdd_real sloppy_add(const gdd_real &a, const gdd_real &b) 
{
	double s, e;

	s = two_sum(a.x, b.x, e);
	e += (a.y + b.y);
	s = quick_two_sum(s, e, e);
	return make_dd(s, e);
}

__inline__ __host__ __device__
gdd_real operator+(const gdd_real &a, const gdd_real &b) 
{
	return sloppy_add(a, b);
}

/*********** Subtractions *********/

__device__
gdd_real operator-(const gdd_real &a, const gdd_real &b) 
{

	double s, e;
	s = two_diff(a.x, b.x, e);
	//return make_dd(s, e);
	e += a.y;
	e -= b.y;
	s = quick_two_sum(s, e, e);
	return make_dd(s, e);

/*
  double s1, s2, t1, t2;
  s1 = two_diff(a.x, b.x, s2);
  t1 = two_diff(a.y, b.y, t2);
  s2 += t1;
  s1 = quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = quick_two_sum(s1, s2, s2);
  return make_dd(s1, s2);
*/
}

/* double-double - double */
__device__
gdd_real operator-(const gdd_real &a, double b) 
{
	double s1, s2;
	s1 = two_diff(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_dd(s1, s2);
}

/* double - double-double */
__device__
gdd_real operator-(double a, const gdd_real &b) {
  double s1, s2;
  s1 = two_diff(a, b.x, s2);
  s2 -= b.y;
  s1 = quick_two_sum(s1, s2, s2);
  return make_dd(s1, s2);
}

/*********** Squaring **********/
__device__
gdd_real sqr(const gdd_real &a) 
{
	double p1, p2;
	double s1, s2;
	p1 = two_sqr(a.x, p2);
	//p2 += (2.0 * a.x * a.y);
        p2 = __dadd_rn(p2,__dmul_rn(__dmul_rn(2.0,a.x), a.y));
	//p2 += (a.y * a.y);
	p2 = __dadd_rn(p2, __dmul_rn(a.y,a.y));
	s1 = quick_two_sum(p1, p2, s2);
	return make_dd(s1, s2);
}

__device__
gdd_real sqr(double a) 
{
	double p1, p2;
	p1 = two_sqr(a, p2);
	return make_dd(p1, p2);
}

/****************** Multiplication ********************/


/* double-double * (2.0 ^ exp) */
__device__
gdd_real ldexp(const gdd_real &a, int exp) 
{
	return make_dd(ldexp(a.x, exp), ldexp(a.y, exp));
}

/* double-double * double,  where double is a power of 2. */
__device__
gdd_real mul_pwr2(const gdd_real &a, double b)
{
	return make_dd(a.x * b, a.y * b);
}

/* double-double * double-double */
__device__
gdd_real operator*(const gdd_real &a, const gdd_real &b)
{
	double p1, p2;

	p1 = two_prod(a.x, b.x, p2);
	p2 += (a.x * b.y + a.y * b.x);
	//p2 += __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x));
        
	p1 = quick_two_sum(p1, p2, p2);

	return make_dd(p1, p2);
}

/* double-double * double */
__device__
gdd_real operator*(const gdd_real &a, double b) 
{
	double p1, p2;

	p1 = two_prod(a.x, b, p2);
	p2 = __dadd_rn(p2,(__dmul_rn(a.y,b)));
	p1 = quick_two_sum(p1, p2, p2);
	return make_dd(p1, p2);
}

/* double * double-double */
__device__
gdd_real operator*(double a, const gdd_real &b) 
{
	return (b * a);
}


/******************* Division *********************/

__device__
gdd_real sloppy_div(const gdd_real &a, const gdd_real &b) 
{
	double s1, s2;
	double q1, q2;
	gdd_real r;

	q1 = a.x / b.x;  /* approximate quotient */

	/* compute  this - q1 * dd */
	r = b * q1;
	s1 = two_diff(a.x, r.x, s2);
	s2 -= r.y;
	s2 += a.y;

	/* get next approximation */
	q2 = (s1 + s2) / b.x;

	/* renormalize */
	r.x = quick_two_sum(q1, q2, r.y);
	return r;
}

/* double-double / double-double */
__device__
gdd_real operator/(const gdd_real &a, const gdd_real &b) 
{
	return sloppy_div(a, b);
}



/* double-double / double */
__device__
gdd_real operator/(const gdd_real &a, double b) {

	double q1, q2;
	double p1, p2;
	double s, e;
	gdd_real r;
 
	q1 = a.x / b;   /* approximate quotient. */

	/* Compute  this - q1 * d */
	p1 = two_prod(q1, b, p2);
	s = two_diff(a.x, p1, e);
	e = e + a.y;
	e = e - p2;
  
	/* get next approximation. */
	q2 = (s + e) / b;

	/* renormalize */
	r.x = quick_two_sum(q1, q2, r.y);

	return r;
}


__host__ __device__
bool is_zero( const gdd_real &a ) 
{
	return (a.x == 0.0);
}

__host__ __device__
bool is_one( const gdd_real &a ) 
{
	return (a.x == 1.0 && a.y == 0.0);
}


/*  this > 0 */
__device__ 
bool is_positive(const gdd_real &a) {
	return (a.x > 0.0);
}

/* this < 0 */
__device__ 
bool is_negative(const gdd_real &a) {
	return (a.x < 0.0);
}

/* Cast to double. */
__device__
double to_double(const gdd_real &a)
{
	return a.x;
}

/************* Comparison ***************/


/* double-double <= double-double */
__host__ __device__
bool operator<=(const gdd_real &a, const gdd_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y <= b.y));
}



/*********** Equality Comparisons ************/
/* double-double == double */
__host__ __device__ bool operator==(const gdd_real &a, double b) {
  return (a.x == b && a.y == 0.0);
}

/* double-double == double-double */
__host__ __device__ bool operator==(const gdd_real &a, const gdd_real &b) {
  return (a.x == b.x && a.y == b.y);
}

/* double == double-double */
__host__ __device__ bool operator==(double a, const gdd_real &b) {
  return (a == b.x && b.y == 0.0);
}

/*********** Greater-Than Comparisons ************/
/* double-double > double */
__host__ __device__ bool operator>(const gdd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y > 0.0));
}

/* double-double > double-double */
__host__ __device__ bool operator>(const gdd_real &a, const gdd_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

/* double > double-double */
__host__ __device__ bool operator>(double a, const gdd_real &b) {
  return (a > b.x || (a == b.x && b.y < 0.0));
}

/*********** Less-Than Comparisons ************/
/* double-double < double */
__host__ __device__ bool operator<(const gdd_real &a, double b) {
  return (a.x < b || (a.x == b && a.y < 0.0));
}

/* double-double < double-double */
__host__ __device__ bool operator<(const gdd_real &a, const gdd_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

/* double < double-double */
__host__ __device__ bool operator<(double a, const gdd_real &b) {
  return (a < b.x || (a == b.x && b.y > 0.0));
}

/*********** Greater-Than-Or-Equal-To Comparisons ************/
/* double-double >= double */
__host__ __device__ bool operator>=(const gdd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

/* double-double >= double-double */
__host__ __device__ bool operator>=(const gdd_real &a, const gdd_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y >= b.y));
}

/* double >= double-double */
//__host__ __device__ bool operator>=(double a, const gdd_real &b) {
//  return (b <= a);
//}

/*********** Less-Than-Or-Equal-To Comparisons ************/
/* double-double <= double */
__host__ __device__ bool operator<=(const gdd_real &a, double b) {
  return (a.x < b || (a.x == b && a.y <= 0.0));
}

/* double >= double-double */
__host__ __device__ bool operator>=(double a, const gdd_real &b) {
  return (b <= a);
}


/* double-double <= double-double */
//__host__ __device__ bool operator<=(const gdd_real &a, const gdd_real &b) {
//  return (a.x[0] < b.x[0] || (a.x[0] == b.x[0] && a.x[1] <= b.x[1]));
//}

/* double <= double-double */
__host__ __device__ bool operator<=(double a, const gdd_real &b) {
  return (b >= a);
}

/*********** Not-Equal-To Comparisons ************/
/* double-double != double */
__host__ __device__ bool operator!=(const gdd_real &a, double b) {
  return (a.x != b || a.y != 0.0);
}

/* double-double != double-double */
__host__ __device__ bool operator!=(const gdd_real &a, const gdd_real &b) {
  return (a.x != b.x || a.y != b.y);
}

/* double != double-double */
__host__ __device__ bool operator!=(double a, const gdd_real &b) {
  return (a != b.x || b.y != 0.0);
}



__device__
gdd_real nint(const gdd_real &a) {
  double hi = nint(a.x);
  double lo;

  if (hi == a.x) {
    /* High word is an integer already.  Round the low word.*/
    lo = nint(a.y);
    
    /* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
    hi = quick_two_sum(hi, lo, lo);
  } else {
    /* High word is not an integer. */
    lo = 0.0;
    if (fabs(hi-a.x) == 0.5 && a.y < 0.0) {
      /* There is a tie in the high word, consult the low word 
         to break the tie. */
      hi -= 1.0;      /* NOTE: This does not cause INEXACT. */
    }
  }

  return make_dd(hi, lo);
}


__device__
gdd_real abs(const gdd_real &a) {
        return (a.x < 0.0) ? negative(a) : a;
}


__device__
gdd_real fabs(const gdd_real &a) {
	return abs(a);
}


/* double / double-double */
__device__
gdd_real operator/(double a, const gdd_real &b) {
	return make_dd(a) / b;
}

__device__
gdd_real inv(const gdd_real &a) {
  return 1.0 / a;
}

#endif /* __GDD_BASIC_CU__ */

