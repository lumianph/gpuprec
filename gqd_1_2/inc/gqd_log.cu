#ifndef __GQD_LOG_CU__
#define __GQD_LOG_CU__

#include "common.cu"


__device__
gqd_real log(const gqd_real &a) {
        if (is_one(a)) {
                return make_qd(0.0);
        }

        //!!!!!!!!!!!!!
        if (a.x <= 0.0) {
                //qd_real::error("(qd_real::log): Non-positive argument.");
                //return qd_real::_nan;
                return make_qd( 0.0 );
        }

        //!!!!!!!!!!!!!!
        if (a.x == 0.0)      {
                //return _inf;
                //TO DO: return an error
                return make_qd( 0.0 );
        }

        gqd_real x = make_qd(log(a.x));  
        
        x = x + a * exp(negative(x)) - 1.0;
        x = x + a * exp(negative(x)) - 1.0;
        x = x + a * exp(negative(x)) - 1.0;

        return x;
}


#endif /* __GQD_LOG_CU__ */
