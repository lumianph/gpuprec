

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <omp.h>
#include "test_util.h"

using namespace std;

void randArray(dd_real* data, const unsigned numElement,
        dd_real low, dd_real high, int seed) {

    assert(high > low);
    srand(seed);
    dd_real band = high - low;

    for (unsigned int i = 0; i < numElement; i++) {
        data[i] = low + ddrand() * band;
    }
}

void randArray(qd_real* data, const unsigned numElement,
        qd_real low, qd_real high, int seed) {

    assert(high > low);
    srand(seed);
    qd_real band = high - low;

    for (unsigned int i = 0; i < numElement; i++) {
        data[i] = low + qdrand() * band;
    }
}

void qd2gqd(const dd_real* dd_data, gdd_real* gdd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        gdd_data[i].x = dd_data[i].x[0];
        gdd_data[i].y = dd_data[i].x[1];
    }
}

void qd2gqd(const qd_real* qd_data, gqd_real* gqd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        gqd_data[i].x = qd_data[i].x[0];
        gqd_data[i].y = qd_data[i].x[1];
        gqd_data[i].z = qd_data[i].x[2];
        gqd_data[i].w = qd_data[i].x[3];
    }
}

void gqd2qd(const gdd_real* gdd_data, dd_real* dd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        dd_data[i].x[0] = gdd_data[i].x;
        dd_data[i].x[1] = gdd_data[i].y;
    }
}

void gqd2qd(const gqd_real* gqd_data, qd_real* qd_data, const unsigned int numElement) {
    for (unsigned int i = 0; i < numElement; i++) {
        qd_data[i].x[0] = gqd_data[i].x;
        qd_data[i].x[1] = gqd_data[i].y;
        qd_data[i].x[2] = gqd_data[i].z;
        qd_data[i].x[3] = gqd_data[i].w;
    }
}

void randArray(gdd_real* data, const unsigned numElement,
        dd_real low, dd_real high, int seed) {
	dd_real* dd_data = new dd_real[numElement];

	randArray(dd_data, numElement, low, high, seed);
	qd2gqd(dd_data, data, numElement);

	delete[] dd_data;
}

int checkTwoArray(const gdd_real* gold, const gdd_real* ref, const int numElement) {
	dd_real* dd_gold = new dd_real[numElement];
	dd_real* dd_ref = new dd_real[numElement];

	gqd2qd(gold, dd_gold, numElement);
	gqd2qd(ref, dd_ref, numElement);

	checkTwoArray(dd_gold, dd_ref, numElement);

	delete[] dd_gold;
	delete[] dd_ref;
}

int checkTwoArray(const gqd_real* gold, const gqd_real* ref, const int numElement) {
	qd_real* qd_gold = new qd_real[numElement];
	qd_real* qd_ref = new qd_real[numElement];

	gqd2qd(gold, qd_gold, numElement);
	gqd2qd(ref, qd_ref, numElement);

	checkTwoArray(qd_gold, qd_ref, numElement);

	delete[] qd_gold;
	delete[] qd_ref;
}

int checkTwoArray(const dd_real* gold, const dd_real* ref, const int numElement) {
    dd_real maxRelError = 0.0;
    dd_real avgRelError = 0.0;
    int maxId = 0;

    for (int i = 0; i < numElement; i++) {
        dd_real relError = abs((gold[i] - ref[i]) / gold[i]);
        avgRelError += (relError / numElement);
        if (relError > maxRelError) {
            maxRelError = relError;
            maxId = i;
        }
    }

    cout << "abs. of max. relative error: " << maxRelError << endl;
    cout << "abs. of avg. relative error: " << avgRelError << endl;
    if (maxRelError > 0.0) {
        cout << "max. relative error elements" << endl;
        cout << "i = " << maxId << endl;
        cout << "gold = " << gold[maxId].to_string() << endl;
        printf("Components(%.16e, %.16e)\n", gold[maxId].x[0], gold[maxId].x[1]);
        cout << "ref  = " << ref[maxId].to_string() << endl;
        printf("Components(%.16e, %.16e)\n", ref[maxId].x[0], ref[maxId].x[1]);

    } else {
        cout << "a sample:" << endl;
        const int i = rand() % numElement;
        cout << "i = " << i << endl;
        cout << "gold = " << gold[i].to_string() << endl;
        printf("Components(%.16e, %.16e)\n", gold[maxId].x[0], gold[maxId].x[1]);
        cout << "ref  = " << ref[i].to_string() << endl;
        printf("Components(%.16e, %.16e)\n", ref[maxId].x[0], ref[maxId].x[1]);
        maxId = i;
    }

    return maxId;
}

int checkTwoArray(const qd_real* gold, const qd_real* ref, const int numElement) {
    qd_real maxRelError = 0.0;
    qd_real avgRelError = 0.0;
    int maxId = 0;

    for (int i = 0; i < numElement; i++) {
        qd_real relError = abs((gold[i] - ref[i]) / gold[i]);
        avgRelError += (relError / numElement);
        if (relError > maxRelError) {
            maxRelError = relError;
            maxId = i;
        }
    }

    cout << "abs. of max. relative error: " << maxRelError << endl;
    cout << "abs. of avg. relative error: " << avgRelError << endl;
    if (maxRelError > 0.0) {
        cout << "max. relative error elements" << endl;
        cout << "i = " << maxId << endl;
        cout << "gold = " << (gold[maxId]).to_string() << endl;
        cout << "ref  = " << (ref[maxId]).to_string() << endl;
        cout << "rel. error = " << (abs((gold[maxId] - ref[maxId]) / gold[maxId])).to_string() << endl;
    } else {
        cout << "a sample:" << endl;
        const int i = rand() % numElement;
        cout << "i = " << i << endl;
        cout << "gold = " << (gold[i]).to_string() << endl;
        cout << "ref  = " << (ref[i]).to_string() << endl;
        maxId = i;
    }

    return maxId;
}

ostream& operator<<(ostream& os, const gqd_real& a) {
	qd_real b(a.x, a.y, a.z, a.w);
	os << b;

	return os;
}

ostream& operator<<(ostream& os, const gdd_real& a) {
	dd_real b(a.x, a.y);
	os << b;

	return os;
}

