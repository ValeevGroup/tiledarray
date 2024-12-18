/***********************  intel_mkl_feature_patch.c  **************************
 * Author:           Agner Fog
 * Date created:     2014-07-30
 * Last modified:    2019-12-29
 * Source URL:       https://www.agner.org/optimize/intel_dispatch_patch.zip
 * Language:         C or C++
 *
 * Description:
 * Patch for Intel Math Kernel Library (MKL) version 14.0 and later, except
 * the Vector Math Library (VML).
 *
 * Example of how to patch Intel's CPU feature dispatcher in order to improve
 * compatibility of Intel function libraries with non-Intel processors.
 * In Windows: Use the static link libraries (*.lib), not the dynamic link
 * librarise (*.DLL).
 * In Linux and Mac: use static linking (*.a) or dynamic linking (*.so).
 *
 * Include this code in your C or C++ program and call intel_mkl_patch();
 * before any call to the MKL functions. You may need to include
 * intel_mkl_cpuid_patch.c as well.
 *
 * Copyright (c) 2014-2019. BSD License 2.0
 ******************************************************************************/
#include <stdint.h>

#ifdef __cplusplus  // use C-style linking
extern "C" {
#endif

// link to MKL libraries
extern int64_t __intel_mkl_feature_indicator;    // CPU feature bits
extern int64_t __intel_mkl_feature_indicator_x;  // CPU feature bits
void __intel_mkl_features_init();  // unfair dispatcher: checks CPU features for
                                   // Intel CPU's only
void __intel_mkl_features_init_x();  // fair dispatcher: checks CPU features
                                     // without discriminating by CPU brand

#ifdef __cplusplus
}  // end of extern "C"
#endif

void intel_mkl_use_fair_dispatch() {
  // force a re-evaluation of the CPU features without discriminating by CPU
  // brand
  __intel_mkl_feature_indicator = 0;
  __intel_mkl_feature_indicator_x = 0;
  __intel_mkl_features_init_x();
  __intel_mkl_feature_indicator = __intel_mkl_feature_indicator_x;
}
