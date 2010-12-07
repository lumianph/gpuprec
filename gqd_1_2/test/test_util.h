#ifndef __GQD_TEST_UTIL_H__
#define __GQD_TEST_UTIL_H__

#include "gqd_type.h"
#include <sys/time.h>
#include <qd/qd_real.h>


void randArray(dd_real* data, const unsigned numElement, 
	       dd_real low, dd_real up, int seed = 0);


void randArray(qd_real* data, const unsigned numElement,
               qd_real low, qd_real up, int seed = 0);

void qd2gqd(dd_real* dd_data, gdd_real* gdd_data, const unsigned int numElement);

void qd2gqd(qd_real* qd_data, gqd_real* gqd_data, const unsigned int numElement);

void gqd2qd(gdd_real* gdd_data, dd_real* dd_data, const unsigned int numElement);

void gqd2qd(gqd_real* gqd_data, qd_real* qd_data, const unsigned int numElement);

int checkTwoArray( const dd_real* gold, const dd_real* ref, const int numElement ); 

int checkTwoArray( const qd_real* gold, const qd_real* ref, const int numElement );

/* timing functions */
inline double getSec( struct timeval tvStart, struct timeval tvEnd ) {
        double tStart = (double)tvStart.tv_sec + 1e-6*tvStart.tv_usec;
        double tEnd = (double)tvEnd.tv_sec + 1e-6*tvEnd.tv_usec;
        return (tEnd - tStart);
}

#define INIT_TIMER struct timeval start, end;
#define START_TIMER gettimeofday(&start, NULL);
#define END_TIMER   gettimeofday(&end, NULL);
#define PRINT_TIMER_SEC(msg) printf("*** %s: %.3f sec ***\n", msg, getSec(start, end));

#endif /* __GQD_TEST_UTIL_H__ */


