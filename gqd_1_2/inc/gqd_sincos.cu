#ifndef __GQD_SIN_COS_CU__
#define __GQD_SIN_COS_CU__

#include "common.cu"


__device__
void sincos_taylor(const gqd_real &a, 
				   gqd_real &sin_a, gqd_real &cos_a) 
{
	const double thresh = 0.5 * _qd_eps * fabs(to_double(a));
	gqd_real p, s, t, x;

	if (is_zero(a)) {
		sin_a.x = sin_a.y = sin_a.z = sin_a.w = 0.0;
		cos_a.x = 1.0;
		cos_a.y = cos_a.z = cos_a.w = 0.0;
		return;
	}

	//x = -sqr(a);
	x = negative( sqr(a) );
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * inv_fact[i];
		s = s + t;
		i = i + 2;
	} while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	sin_a = s;
	cos_a = sqrt(1.0 - sqr(s));
}


__device__
gqd_real sin_taylor(const gqd_real &a) {
	const double thresh = 0.5 * _qd_eps * fabs(to_double(a));
	gqd_real p, s, t, x;

	if (is_zero(a)) {
		//return make_qd(0.0);
		s.x = s.y = s.z = s.w = 0.0;
		return s;
	}

	//x = -sqr(a);
	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}


__device__
gqd_real cos_taylor(const gqd_real &a) {
	const double thresh = 0.5 * _qd_eps;
	gqd_real p, s, t, x;

	if (is_zero(a)) {
		//return make_qd(1.0);
		s.x = 1.0;
		s.y = s.z = s.w = 0.0;
		return s;
	}

	//x = -sqr(a);
	x = negative(sqr(a));
	s = 1.0 + mul_pwr2(x, 0.5);
	p = x;
	int i = 1;
	do {
		p = p * x;
		t = p * inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}


__device__
gqd_real sin(const gqd_real &a) {

	gqd_real z, r;
	if (is_zero(a)) {
		//return make_qd(0.0);
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	// approximately reduce modulo 2*pi
	z = nint(a / _qd_2pi);
	r = a - _qd_2pi * z;

	// approximately reduce modulo pi/2 and then modulo pi/1024
	double q = floor(r.x / _qd_pi2.x + 0.5);
	gqd_real t = r - _qd_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qd_pi1024.x + 0.5);
	t = t - _qd_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//gqd_real::error("(gqd_real::sin): Cannot reduce modulo pi/2.");
		//return gqd_real::_nan;
		//return make_qd(0.0);
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	if (abs_k > 256) {
		//gqd_real::error("(gqd_real::sin): Cannot reduce modulo pi/1024.");
		//return gqd_real::_nan;
		//return make_qd( 0.0 );
		r.x = r.y = r.z = r.w = 0.0;
		return r;
	}

	if (k == 0) {
		switch (j) {
	  case 0:
		  return sin_taylor(t);
	  case 1:
		  return cos_taylor(t);
	  case -1:
		  return negative(cos_taylor(t));
	  default:
		  return negative(sin_taylor(t));
		}
	}

	//gqd_real sin_t, cos_t;
	//gqd_real u = d_cos_table[abs_k-1];
	//gqd_real v = d_sin_table[abs_k-1]; 
	//sincos_taylor(t, sin_t, cos_t);
	///use z and r again to avoid allocate additional memory
	///z = sin_t, r = cos_t
	sincos_taylor( t, z, r );

	if (j == 0) {
		z = d_cos_table[abs_k-1] * z;
		r = d_sin_table[abs_k-1] * r;
		if (k > 0) {
			//z = d_cos_table[abs_k-1] * z;
			//r = d_sin_table[abs_k-1] * r;
			return  z + r;
		} else {
			//z = d_cos_table[abs_k-1] * z;
			//r = d_sin_table[abs_k-1] * r;
			return z - r;
		}
	} else if (j == 1) {
		r = d_cos_table[abs_k-1] * r;
		z = d_sin_table[abs_k-1] * z;
		if (k > 0) {
			//r = d_cos_table[abs_k-1] * r;
			//z = d_sin_table[abs_k-1] * z;
			return r - z;
		} else {
			//r = d_cos_table[abs_k-1] * r;
			//z = d_sin_table[abs_k-1] * z;
			return r + z;
		}
	} else if (j == -1) {
		z = d_sin_table[abs_k-1] * z;
		r = d_cos_table[abs_k-1] * r;
		if (k > 0) {
			//z = d_sin_table[abs_k-1] * z;
			//r = d_cos_table[abs_k-1] * r;
			return z - r;
		} else {
			//r = negative(d_cos_table[abs_k-1]) * r;
			//r = (d_cos_table[abs_k-1]) * r;
			r.x = -r.x;
			r.y = -r.y;
			r.z = -r.z;
			r.w = -r.w;
			//z = d_sin_table[abs_k-1] * z;
			return r - z;
		}
	} else {
		r = d_sin_table[abs_k-1] * r ;
		z = d_cos_table[abs_k-1] * z;
		if (k > 0) {
			//z = negative(d_cos_table[abs_k-1]) * z;
			//z = d_cos_table[abs_k-1] * z;
			z.x = -z.x;
			z.y = -z.y;
			z.z = -z.z;
			z.w = -z.w;
			//r = d_sin_table[abs_k-1] * r;
			return z - r;
		} else {
			//r = d_sin_table[abs_k-1] * r ;
			//z = d_cos_table[abs_k-1] * z;
			return r - z;
		}
	}
}


__device__
gqd_real cos(const gqd_real &a) {
	if (is_zero(a)) {
		return make_qd(1.0);
	}

	// approximately reduce modulo 2*pi
	gqd_real z = nint(a / _qd_2pi);
	gqd_real r = a - _qd_2pi * z;

	// approximately reduce modulo pi/2 and then modulo pi/1024
	double q = floor(r.x / _qd_pi2.x + 0.5);
	gqd_real t = r - _qd_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qd_pi1024.x + 0.5);
	t = t - _qd_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//qd_real::error("(qd_real::cos): Cannot reduce modulo pi/2.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	if (abs_k > 256) {
		//qd_real::error("(qd_real::cos): Cannot reduce modulo pi/1024.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	if (k == 0) {
		switch (j) {
	  		case 0:
		  		return cos_taylor(t);
			  case 1:
				  return negative(sin_taylor(t));
			  case -1:
				  return sin_taylor(t);
			  default:
				  return negative(cos_taylor(t));
		}
	}

	gqd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gqd_real u = d_cos_table[abs_k - 1];
	gqd_real v = d_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) {
			r = u * cos_t - v * sin_t;
		} else {
			r = u * cos_t + v * sin_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			r = negative(u * sin_t) - v * cos_t;
		} else {
			r = v * cos_t - u * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			r = u * sin_t + v * cos_t;
		} else {
			r = u * sin_t - v * cos_t;
		}
	} else {
		if (k > 0) {
			r = v * sin_t - u * cos_t;
		} else {
			r = negative(u * cos_t) - v * sin_t;
		}
	}

	return r;
}


__device__
void sincos(const gqd_real &a, gqd_real &sin_a, gqd_real &cos_a) {

	if (is_zero(a)) {
		sin_a = make_qd(0.0);
		cos_a = make_qd(1.0);
		return;
	}

	// approximately reduce by 2*pi
	gqd_real z = nint(a / _qd_2pi);
	gqd_real t = a - _qd_2pi * z;

	// approximately reduce by pi/2 and then by pi/1024.
	double q = floor(t.x / _qd_pi2.x + 0.5);
	t = t - _qd_pi2 * q;
	int j = (int)(q);
	q = floor(t.x / _qd_pi1024.x + 0.5);
	t = t - _qd_pi1024 * q;
	int k = (int)(q);
	int abs_k = abs(k);

	if (j < -2 || j > 2) {
		//qd_real::error("(qd_real::sincos): Cannot reduce modulo pi/2.");
		//cos_a = sin_a = qd_real::_nan;
		cos_a = sin_a = make_qd(0.0);
		return;
	}

	if (abs_k > 256) {
		//qd_real::error("(qd_real::sincos): Cannot reduce modulo pi/1024.");
		//cos_a = sin_a = qd_real::_nan;
		cos_a = sin_a = make_qd(0.0);
		return;
	}

	gqd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if (j == 0) {
			sin_a = sin_t;
			cos_a = cos_t;
		} else if (j == 1) {
			sin_a = cos_t;
			cos_a = negative(sin_t);
		} else if (j == -1) {
			sin_a = negative(cos_t);
			cos_a = sin_t;
		} else {
			sin_a = negative(sin_t);
			cos_a = negative(cos_t);
		}
		return;
	}

	gqd_real u = d_cos_table[abs_k - 1];
	gqd_real v = d_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) {
			sin_a = u * sin_t + v * cos_t;
			cos_a = u * cos_t - v * sin_t;
		} else {
			sin_a = u * sin_t - v * cos_t;
			cos_a = u * cos_t + v * sin_t;
		}
	} else if (j == 1) {
		if (k > 0) {
			cos_a = negative(u * sin_t) - v * cos_t;
			sin_a = u * cos_t - v * sin_t;
		} else {
			cos_a = v * cos_t - u * sin_t;
			sin_a = u * cos_t + v * sin_t;
		}
	} else if (j == -1) {
		if (k > 0) {
			cos_a = u * sin_t + v * cos_t;
			sin_a =  v * sin_t - u * cos_t;
		} else {
			cos_a = u * sin_t - v * cos_t;
			sin_a = negative(u * cos_t) - v * sin_t;
		}
	} else {
		if (k > 0) {
			sin_a = negative(u * sin_t) - v * cos_t;
			cos_a = v * sin_t - u * cos_t;
		} else {
			sin_a = v * cos_t - u * sin_t;
			cos_a = negative(u * cos_t) - v * sin_t;
		}
	}
}


__device__
gqd_real tan(const gqd_real &a) {
  gqd_real s, c;
  sincos(a, s, c);
  return s/c;
}

#ifdef ALL_MATH	

__device__
gqd_real atan2(const gqd_real &y, const gqd_real &x) {

	if (is_zero(x)) {

		if (is_zero(y)) {
			// Both x and y is zero. 
			//qd_real::error("(qd_real::atan2): Both arguments zero.");
			//return qd_real::_nan;
			return make_qd(0.0);
		}

		return (is_positive(y)) ? _qd_pi2 : negative(_qd_pi2);
	} else if (is_zero(y)) {
		return (is_positive(x)) ? make_qd(0.0) : _qd_pi;
	}

	if (x == y) {
		return (is_positive(y)) ? _qd_pi4 : negative(_qd_3pi4);
	}

	if (x == negative(y)) {
		return (is_positive(y)) ? _qd_3pi4 : negative(_qd_pi4);
	}

	gqd_real r = sqrt(sqr(x) + sqr(y));
	gqd_real xx = x / r;
	gqd_real yy = y / r;

	gqd_real z = make_qd(atan2(to_double(y), to_double(x)));
	gqd_real sin_z, cos_z;

	if (abs(xx.x) > abs(yy.x)) {
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
		sincos(z, sin_z, cos_z);
		z = z + (yy - sin_z) / cos_z;
	} else {
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
		sincos(z, sin_z, cos_z);
		z = z - (xx - cos_z) / sin_z;
	}

	return z;
}


__device__
gqd_real atan(const gqd_real &a) {
	return atan2(a, make_qd(1.0));
}


__device__
gqd_real asin(const gqd_real &a) {
	gqd_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//qd_real::error("(qd_real::asin): Argument out of domain.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? _qd_pi2 : negative(_qd_pi2);
	}

	return atan2(a, sqrt(1.0 - sqr(a)));
}


__device__
gqd_real acos(const gqd_real &a) {
	gqd_real abs_a = abs(a);

	if (abs_a > 1.0) {
		//qd_real::error("(qd_real::acos): Argument out of domain.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	if (is_one(abs_a)) {
		return (is_positive(a)) ? make_qd(0.0) : _qd_pi;
	}

	return atan2(sqrt(1.0 - sqr(a)), a);
}


__device__
gqd_real sinh(const gqd_real &a) {
	if (is_zero(a)) {
		return make_qd(0.0);
	}

	if (abs(a) > 0.05) {
		gqd_real ea = exp(a);
		return mul_pwr2(ea - inv(ea), 0.5);
	}

	gqd_real s = a;
	gqd_real t = a;
	gqd_real r = sqr(t);
	double m = 1.0;
	double thresh = abs(to_double(a) * _qd_eps);

	do {
		m = m + 2.0;
		t = (t*r);
		t = t/((m-1) * m);

		s = s + t;
	} while (abs(t) > thresh);

	return s;
}


__device__
gqd_real cosh(const gqd_real &a) {
	if (is_zero(a)) {
		return make_qd(1.0);
	}

	gqd_real ea = exp(a);
	return mul_pwr2(ea + inv(ea), 0.5);
}


__device__
gqd_real tanh(const gqd_real &a) {
	if (is_zero(a)) {
		return make_qd(0.0);
	}

	if (abs(to_double(a)) > 0.05) {
		gqd_real ea = exp(a);
		gqd_real inv_ea = inv(ea);
		return (ea - inv_ea) / (ea + inv_ea);
	} else {
		gqd_real s, c;
		s = sinh(a);
		c = sqrt(1.0 + sqr(s));
		return s / c;
	}
}


__device__
void sincosh(const gqd_real &a, gqd_real &s, gqd_real &c) {
	if (abs(to_double(a)) <= 0.05) {
		s = sinh(a);
		c = sqrt(1.0 + sqr(s));
	} else {
		gqd_real ea = exp(a);
		gqd_real inv_ea = inv(ea);
		s = mul_pwr2(ea - inv_ea, 0.5);
		c = mul_pwr2(ea + inv_ea, 0.5);
	}
}


__device__
gqd_real asinh(const gqd_real &a) {
	return log(a + sqrt(sqr(a) + 1.0));
}


__device__
gqd_real acosh(const gqd_real &a) {
	if (a < 1.0) {
		///qd_real::error("(qd_real::acosh): Argument out of domain.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	return log(a + sqrt(sqr(a) - 1.0));
}


__device__
gqd_real atanh(const gqd_real &a) {
	if (abs(a) >= 1.0) {
		//qd_real::error("(qd_real::atanh): Argument out of domain.");
		//return qd_real::_nan;
		return make_qd(0.0);
	}

	return mul_pwr2(log((1.0 + a) / (1.0 - a)), 0.5);
}


#endif /* ALL_MATH */


#endif /* __GQD_SIN_COS_CU__ */


