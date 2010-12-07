#ifndef __GDD_EXP_CU__
#define __GDD_EXP_CU__

#include "common.cu"

#define INV_K (1.0/512.0) 



//the completed version with additional branches for parameter checking
/*
__device__
gdd_real exp(const gdd_real &a) {

        const double k = 512.0;
        const double inv_k = 1.0 / k;

        if (a.x <= -709.0)
                return make_dd(0.0);

        if (a.x >=  709.0)
                return make_dd(0.0);
	        //TODO: return dd_real::_inf;

        if (is_zero(a))
                return make_dd(1.0);

        if (is_one(a))
                return _dd_e;

        double m = floor(a.x / _dd_log2.x + 0.5);
        gdd_real r = mul_pwr2(a - _dd_log2 * m, inv_k);
        gdd_real s, t, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        t = p * dd_inv_fact[0];
        int i = 0;
        do {
                s = s + t;
                p = p * r;
                t = p * dd_inv_fact[++i];
        } while ((fabs(to_double(t)) > inv_k * _dd_eps) && (i < 5));

        s = s + t;

        for( int i = 0; i < 9; i++ )
        {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }

        s = s + 1.0;

        return ldexp(s, int(m));
}
*/

__device__
gdd_real exp(const gdd_real &a) {

        double m = floor(a.x / _dd_log2.x + 0.5);
        gdd_real r = mul_pwr2(a - _dd_log2 * m, INV_K);
        gdd_real s, t, p;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        p = p * r;
        t = p * dd_inv_fact[0];
        int i = 0;
        do {
                s = s + t;
                p = p * r;
                t = p * dd_inv_fact[++i];
        } while ((fabs(to_double(t)) > INV_K * _dd_eps) && (i < 5));
        s = s + t;
	
	for( int i = 0; i < 9; i++ ) {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }
        s = s + 1.0;

        return ldexp(s, int(m));
}


#endif /* __GDD_EXP_CU__ */


