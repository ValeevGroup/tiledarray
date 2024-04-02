/***********************  intel_mkl_cpuid_patch.c  **************************
 * Author:           Agner Fog
 * Date created:     2019-12-29
 * Source URL:       https://www.agner.org/optimize/intel_dispatch_patch.zip
 * Language:         C or C++
 *
 * Description:
 * Patch for Intel Math Kernel Library (MKL) version 14.0 and later, except
 * the Vector Math Library (VML).
 *
 * Example of how to override Intel's CPU feature dispatcher in order to improve
 * compatibility of Intel function libraries with non-Intel processors.
 *
 * Include this code in your C or C++ program and make sure it is linked before
 * any Intel libraries. You may need to include intel_mkl_feature_patch.c as
 *well.
 *
 * Copyright (c) 2019. BSD License 2.0
 ******************************************************************************/
#include <stdint.h>

#ifdef __cplusplus  // use C-style linking
extern "C" {
#endif

// detect if Intel CPU
int mkl_serv_intel_cpu() { return 1; }

// detect if Intel CPU
int mkl_serv_intel_cpu_true() { return 1; }

int mkl_serv_cpuhaspnr_true() { return 1; }

int mkl_serv_cpuhaspnr() { return 1; }

int mkl_serv_cpuhasnhm() { return 1; }

int mkl_serv_cpuisbulldozer() { return 0; }

int mkl_serv_cpuiszen() { return 0; }

int mkl_serv_cpuisatomsse4_2() { return 0; }

int mkl_serv_cpuisatomssse3() { return 0; }

int mkl_serv_cpuisitbarcelona() { return 0; }

int mkl_serv_cpuisskl() { return 0; }

int mkl_serv_cpuisknm() { return 0; }

int mkl_serv_cpuisclx() { return 0; }

int mkl_serv_get_microarchitecture() {
  // I don't know what this number means
  return 33;
}

#ifdef __cplusplus
}  // end of extern "C"
#endif
