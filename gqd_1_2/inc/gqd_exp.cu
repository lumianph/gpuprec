#ifndef __GQD_EXP_CU__
#define __GQD_EXP_CU__

#include "common.cu"

__device__
gqd_real exp( const gqd_real &a ) {
        gqd_real r;
        const double k = ldexp(1.0, 16);
        const double inv_k = 1.0 / k;

        if (a.x <= -709.0) {
                //return make_qd(0.0);
                r.x = r.y = r.z = r.w = 0.0;
                return r;
        }

        //!!!!!!!!!!!!!!
        if (a.x >=  709.0) {
                //return make_qd(0.0);
                //return qd_inf;
                r.x = r.y = r.z = r.w = 0.0;
                return r;
        }

        if (is_zero(a)) {
                //return make_qd(1.0);
                r.x = 1.0;
                r.y = r.z = r.w = 0.0;
                return r;
        }

        if (is_one(a)) {
                //return _qd_e;
                r.x = 2.718281828459045091e+00;
                r.y = 1.445646891729250158e-16;
                r.z = -2.127717108038176765e-33;
                r.w = 1.515630159841218954e-49;
                
                return r;
        }

        double m = floor(a.x/_qd_log2.x + 0.5);
        r = mul_pwr2(a - _qd_log2 * m, inv_k);
        gqd_real s, p, t;
        double thresh = inv_k * _qd_eps;

        p = sqr(r);
        s = r + mul_pwr2(p, 0.5);
        int i = 0;
        do 
        {
                p = p * r;
                t = p * inv_fact[i++];
                s = s + t;
        } while ( (fabs(to_double(t)) > thresh) && (i < 9) );

        for( int i = 0; i < 16; i++ )
        {
                s = mul_pwr2(s, 2.0) + sqr(s);
        }
        s = s + 1.0;

        return ldexp(s, int(m));
}


#endif /* __GQD_EXP_CU__ */


