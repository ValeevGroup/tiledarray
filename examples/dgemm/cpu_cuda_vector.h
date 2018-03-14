#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include "platform.h"

// thrust::device_vector::data() returns a proxy, provide an overload for std::data() to provide raw ptr
namespace thrust {
  template<typename T, typename Alloc>
  const T* data (const thrust::device_vector<T, Alloc>& dev_vec) {
    return thrust::raw_pointer_cast(dev_vec.data());
  }
  template<typename T, typename Alloc>
  T* data (thrust::device_vector<T, Alloc>& dev_vec) {
    return thrust::raw_pointer_cast(dev_vec.data());
  }

  // this must be instantiated in a .cu file
  template <typename T, typename Alloc>
  void resize(thrust::device_vector<T, Alloc>& dev_vec, size_t size);
}  // namespace thrust

#include <btas/array_adaptor.h>

namespace TiledArray {

/// \brief a vector that lives on either host or device side, or both

/// \tparam T the type of values this vector holds
/// \tparam HostAlloc The allocator type used for host data
/// \tparam DeviceAlloc The allocator type used for device data
template <typename T, typename HostAlloc = std::allocator<T>,
    typename DeviceAlloc = thrust::device_malloc_allocator<T>>
class cpu_cuda_vector {
  public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef typename thrust::host_vector<T, HostAlloc>::size_type size_type;
    typedef typename thrust::host_vector<T, HostAlloc>::difference_type difference_type;
    typedef typename thrust::host_vector<T, HostAlloc>::iterator iterator;
    typedef typename thrust::host_vector<T, HostAlloc>::const_iterator const_iterator;

    enum class state {
      none = 0x00, host = 0x01, device = 0x10, all = 0x11
    };

    /// creates an empty vector
    cpu_cuda_vector() : state_(state::host) {}
    /// creates a vector with \c size elements
    /// \param st the target state of the vector
    cpu_cuda_vector(size_type size, state st = state::host) :
      host_vec_(static_cast<int>(st) & static_cast<int>(state::host) ? size : 0),
      state_(st)
    {
      if (static_cast<int>(st) & static_cast<int>(state::device))
        thrust::resize(device_vec_, size);
    }
    /// creates a vector with \c size elements filled with \c value
    /// \param value the value used to fill the vector
    /// \param st the target state of the vector
    cpu_cuda_vector(size_type size, T value, state st = state::host) :
      host_vec_(static_cast<int>(st) & static_cast<int>(state::host) ? size : 0, value),
      device_vec_(static_cast<int>(st) & static_cast<int>(state::device) ? size : 0, value),
      state_(st)
    {}
    /// initializes on host side from an iterator range
    /// \tparam RandomAccessIterator
    template <typename RandomAccessIterator>
    cpu_cuda_vector(RandomAccessIterator begin, RandomAccessIterator end) :
      host_vec_(begin, end),
      state_(state::host)
    {
    }

    size_type size() const {
      if (on_host())
        return host_vec_.size();
      if (on_device())
        return device_vec_.size();
    }

    void resize(size_type new_size) {
      if (on_host())
        host_vec_.resize(new_size);
      if (on_device()) {
        //device_vec_.resize(new_size);
        assert(false);
      }
    }

    /// moves the data from the host to the device (even if it's there)
    void to_device() const {
      assert(on_host());
      device_vec_ = host_vec_;
      state_ = state::all;
    }
    /// moves the data from the device to the host (even if it's there)
    void to_host() const {
      assert(on_device());
      host_vec_ = device_vec_;
      state_ = state::all;
    }

    const T* host_data() const {
      assert(on_host());
      return host_vec_.data();
    }
    T* host_data() {
      assert(on_host());
      state_ = state::host;
      return host_vec_.data();
    }
    const T* device_data() const {
      assert(on_device());
      return thrust::raw_pointer_cast(device_vec_.data());
    }
    T* device_data() {
      assert(on_device());
      state_ = state::device;
      return thrust::raw_pointer_cast(device_vec_.data());
    }

    const T* data() const {
      return host_data();
    }
    T* data() {
      return host_data();
    }

    iterator begin() {
      assert(on_host());
      return std::begin(host_vec_);
    }
    const_iterator begin() const {
      assert(on_host());
      return std::cbegin(host_vec_);
    }
    const_iterator cbegin() const {
      assert(on_host());
      return std::cbegin(host_vec_);
    }
    iterator end() {
      assert(on_host());
      return std::end(host_vec_);
    }
    const_iterator end() const {
      assert(on_host());
      return std::cend(host_vec_);
    }
    const_iterator cend() const {
      assert(on_host());
      return std::cend(host_vec_);
    }

    bool on_host() const {
      return static_cast<int>(state_) & static_cast<int>(state::host);
    }
    bool on_device() const {
      return static_cast<int>(state_) & static_cast<int>(state::device);
    }

  private:
    mutable thrust::host_vector<T, HostAlloc> host_vec_;
    mutable thrust::device_vector<T, DeviceAlloc> device_vec_;
    mutable state state_;

};

extern template
class cpu_cuda_vector<double>;
extern template
class cpu_cuda_vector<float>;

template <MemorySpace Space, typename T, typename HostAlloc, typename DeviceAlloc>
bool in_memory_space(const cpu_cuda_vector<T, HostAlloc, DeviceAlloc>& vec) noexcept {
  return
      (vec.on_host() && overlap(MemorySpace::CPU, Space)) ||
      (vec.on_device() && overlap(MemorySpace::CUDA, Space));
}

}  // namespace TiledArray

namespace madness {
namespace archive {

// forward decls
template<class Archive, typename T> struct ArchiveLoadImpl;
template<class Archive, typename T> struct ArchiveStoreImpl;

template<class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::cpu_cuda_vector<T> > {
  static inline void load(const Archive& ar, TiledArray::cpu_cuda_vector<T>& x) {
    typename TiledArray::cpu_cuda_vector<T>::size_type n;
    ar & n;
    x.resize(n);
    for (auto& xi : x)
      ar & xi;
  }
};

template<class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::cpu_cuda_vector<T> > {
  static inline void store(const Archive& ar, const TiledArray::cpu_cuda_vector<T>& x) {
    ar & x.size();
    for (const auto& xi : x)
      ar & xi;
  }
};

}
}

