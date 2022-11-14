//
// Created by Eduard Valeyev on 10/18/22.
//

#if defined(__x86_64__)
#if defined(__AVX__)
#define PREFERRED_ALIGN_SIZE 32
#elif defined(__AVX512F__)
#define PREFERRED_ALIGN_SIZE 64
#else  // 64-bit x86 should have SSE
#define PREFERRED_ALIGN_SIZE 16
#endif
#elif (defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || \
       defined(_M_ARM64))
#define PREFERRED_ALIGN_SIZE 16
#elif defined(__VECTOR4DOUBLE__)
#define PREFERRED_ALIGN_SIZE 32
#endif

// else: default to typical cache line size
#ifndef PREFERRED_ALIGN_SIZE
#define PREFERRED_ALIGN_SIZE 64
#endif

/* Preferred align size, in bytes. */
const char info_align_size[] = {
    /* clang-format off */
  'I', 'N', 'F', 'O', ':', 'a', 'l', 'i', 'g', 'n', '_', 's', 'i', 'z',
  'e', '[', ('0' + ((PREFERRED_ALIGN_SIZE / 10) % 10)), ('0' + (PREFERRED_ALIGN_SIZE % 10)), ']',
  '\0'
    /* clang-format on */
};

int main(int argc, char* argv[]) {
  int require = 0;
  require += info_align_size[argc];
  (void)argv;
  return require;
}
