#ifndef TILEDARRAY_UTIL_THREADS_H__INCLUDED
#define TILEDARRAY_UTIL_THREADS_H__INCLUDED

namespace TiledArray {

  static int max_threads = 1;

  int get_num_threads();
  void set_num_threads(int);

  struct scope_num_threads {
    explicit scope_num_threads(int n) {
      n_ = get_num_threads();
      set_num_threads(n);
    }
    ~scope_num_threads() {
      set_num_threads(n_);
    }
    scope_num_threads(const scope_num_threads&) = delete;
  private:
    int n_ = 0;
  };

#define TA_MAX_THREADS                                  \
  TiledArray::scope_num_threads                         \
  ta_scope_num_threads(TiledArray::max_threads)

};

#endif  // TILEDARRAY_UTIL_THREADS_H__INCLUDED
