/***********************  intel_cpu_feature_patch.c  **************************
 * Author:           Agner Fog
 * Date created:     2014-07-30
 * Last modified:    2019-12-29
 * Source URL:       https://www.agner.org/optimize/intel_dispatch_patch.zip
 * Language:         C or C++
 *
 * Description:
 * Patch for Intel compiler version 13.0 and later, including the general
 * libraries, LIBM and SVML, but not MKL and VML.
 *
 * Example of how to patch Intel's CPU feature dispatcher in order to improve
 * compatibility of generated code with non-Intel processors.
 * In Windows: Use the static link libraries (*.lib), not the dynamic link
 * librarise (*.DLL).
 * In Linux and Mac: use static linking (*.a) or dynamic linking (*.so).
 *
 * Include this code in your C or C++ program and call intel_cpu_patch();
 * before any call to the library functions.
 *
 * Copyright (c) 2014-2019. BSD License 2.0
 ******************************************************************************/
#include <stdint.h>

#ifdef __cplusplus  // use C-style linking
extern "C" {
#endif

// link to Intel libraries
extern int64_t __intel_cpu_feature_indicator;    // CPU feature bits
extern int64_t __intel_cpu_feature_indicator_x;  // CPU feature bits
void __intel_cpu_features_init();  // unfair dispatcher: checks CPU features for
                                   // Intel CPU's only
void __intel_cpu_features_init_x();  // fair dispatcher: checks CPU features
                                     // without discriminating by CPU brand

#ifdef __cplusplus
}  // end of extern "C"
#endif

void intel_cpu_patch() {
  // force a re-evaluation of the CPU features without discriminating by CPU
  // brand
  __intel_cpu_feature_indicator = 0;
  __intel_cpu_feature_indicator_x = 0;
  __intel_cpu_features_init_x();
  __intel_cpu_feature_indicator = __intel_cpu_feature_indicator_x;
}
