#include <cstdlib>
#include <cstdio>

#include "gqd.cu"
#include "test_util.h"


using namespace std;

__global__
void compute_kernel(const gdd_real* d_in1, const gdd_real* d_in2,
        const unsigned int numElement,
        gdd_real* d_out) {
    const unsigned numTotalThread = NUM_TOTAL_THREAD;
    const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

    for (unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
        // Use the new double-double type to do the computation
        // You can change to other operators for testing
        d_out[i] = d_in1[i] + d_in2[i];
    }
}

int main() {
    // CPU FPU fix, this is necessary for the qd library
    unsigned int old_cw;
    fpu_fix_start(&old_cw);

    // Turn on the gqd library
    GDDStart();

    // Generate random double-double numbers on the CPU for testing
    const int numElement = 16;
    dd_real* dd_in1 = new dd_real[numElement];
    dd_real* dd_in2 = new dd_real[numElement];
    dd_real low = "-1.0";
    dd_real high = "1.0";
    randArray(dd_in1, numElement, low, high, 777);
    randArray(dd_in2, numElement, low, high, 888);

    // Convert dd_readl to gdd_real, note that the data is still on the host
    gdd_real* gdd_in1 = new gdd_real[numElement];
    gdd_real* gdd_in2 = new gdd_real[numElement];
    qd2gqd(dd_in1, gdd_in1, numElement);
    qd2gqd(dd_in2, gdd_in2, numElement);

    // Allocate memory and copy data to the GPU device
    gdd_real *d_in1, *d_in2, *d_out;
    GPUMALLOC((void**) &d_in1, sizeof (gdd_real) * numElement);
    GPUMALLOC((void**) &d_in2, sizeof (gdd_real) * numElement);
    GPUMALLOC((void**) &d_out, sizeof (gdd_real) * numElement);
    TOGPU(d_in1, gdd_in1, sizeof (gdd_real) * numElement);
    TOGPU(d_in2, gdd_in2, sizeof (gdd_real) * numElement);

    // Call the device kernel to do the computation
    compute_kernel << <128, 128 >> >(d_in1, d_in2, numElement, d_out);
    getLastCudaError("add_kernel");

    // Copy GPU result back to the host, note that still in gdd_real type
    gdd_real* gdd_out = new gdd_real[numElement];
    FROMGPU(gdd_out, d_out, sizeof (gdd_real) * numElement);

    // Convert gdd to dd_real for debug, gdd_real -> dd_real
    dd_real* ref_out = new dd_real[numElement];
    gqd2qd(gdd_out, ref_out, numElement);

    // Call the qd library for verification
    dd_real* gold_out = new dd_real[numElement];
    for (int i = 0; i < numElement; i += 1) {
        gold_out[i] = dd_in1[i] + dd_in2[i];
    }

    // Show both results from the GPU and CPU for investigation
    for (int i = 0; i < numElement; i += 1) {
        cout << "ref:  " << ref_out[i].to_string() << endl;
        cout << "gold: " << gold_out[i].to_string() << endl;
        cout << "---------------------------------" << endl;
    }

    // Memory cleanup
    delete[] dd_in1;
    delete[] dd_in2;
    delete[] gdd_in1;
    delete[] gdd_in2;
    delete[] ref_out;
    delete[] gold_out;
    GPUFREE(d_in1);
    GPUFREE(d_in2);
    GPUFREE(d_out);


    // Turn off the gqd library
    GDDEnd();

    // Exit CPU FPU fix
    fpu_fix_end(&old_cw);

    return EXIT_SUCCESS;
}

