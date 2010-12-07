#ifndef __GQD_TEST_H__
#define __GQD_TEST_H__

#include "gqd_type.h"
#include "test_common.h"
#include "test_util.h"

float device_basic(gdd_real* h_a, gdd_real* h_b, gdd_real* h_out, const unsigned int numElement,
			OPERATOR op = ADD,
                        const unsigned int numBlock = 128,
                        const unsigned int numThread = 128);


float device_basic(gqd_real* h_a, gqd_real* h_b, gqd_real* h_out, const unsigned int numElement,
                        OPERATOR op = ADD,
                        const unsigned int numBlock = 128,
                        const unsigned int numThread = 128);

float device_math(gdd_real* h_in, const unsigned int numElement, gdd_real* h_out,
                   MATH math, const unsigned int numBlock = 128, const unsigned int numThread = 128);


float device_math(gqd_real* h_in, const unsigned int numElement, gqd_real* h_out,
                   MATH math, const unsigned int numBlock = 128, const unsigned int numThread = 128);


float device_defined(gdd_real* h_in, const unsigned int numElement, gdd_real* h_out, 
			const unsigned int numBlock = 128, const unsigned int numThread = 128);

float device_defined(gqd_real* h_in, const unsigned int numElement, gqd_real* h_out, 
                        const unsigned int numBlock = 128, const unsigned int numThread = 128);


void device_qrsmap(const unsigned int N, const int numBlock = 2400, const int numThread = 64);

#endif /* __GQD_TEST_H__ */


