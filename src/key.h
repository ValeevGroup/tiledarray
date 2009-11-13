#ifndef TILEDARRAY_KEY_H__INCLUDED
#define TILEDARRAY_KEY_H__INCLUDED

#include <error.h>

namespace TiledArray {
  namespace detail {

    template<typename Key1, typename Key2>
    class Key;
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator ==(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator ==(const Key2&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator !=(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator !=(const Key2&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator <(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator <(const Key2&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator <=(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator <=(const Key2&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator >(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator >(const Key2&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>&, const Key1&);
    template<typename Key1, typename Key2>
    bool operator >=(const Key1&, const Key<Key1, Key2>&);
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>&, const Key2&);
    template<typename Key1, typename Key2>
    bool operator >=(const Key2&, const Key<Key1, Key2>&);

    /// Key class that holds two arbitrary key types.

    /// Contains two arbitrary key values. It provides methods of comparing the
    /// two keys with the appropriate type. Both key values must be set.
    template<typename Key1, typename Key2>
    class Key {
    public:
      typedef Key<Key1, Key2> Key_;
      typedef Key1 key1_type;
      typedef Key2 key2_type;

      /// Default constructor
      Key() : k1_(), k2_(), k_(0) { }

      /// Key pair constructor
      Key(const key1_type& k1, const key2_type& k2) : k1_(k1), k2_(k2), k_(3) { }

      /// Key1 constructor
      explicit Key(const key1_type& k1) : k1_(k1), k2_(), k_(1) { }

      /// Key2 constructor
      explicit Key(const key2_type& k2) : k1_(), k2_(k2), k_(2) { }

      /// Copy constructor
      Key(const Key_& other) : k1_(other.k1_), k2_(other.k2_), k_(other.k_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor for key 2
      Key(const key1_type& k1, key2_type&& k2) : k1_(k1), k2_(std::move(k2)), k_(3) { }

      /// Move constructor for key 1
      Key(key1_type&& k1, const key2_type& k2) : k1_(std::move(k1)), k2_(k2), k_(3) { }

      /// Move constructor for key 1 and key 2
      Key(key1_type&& k1, key2_type&& k2) : k1_(std::move(k1)), k2_(std::move(k2)), k_(3) { }

      /// Key1 move constructor
      Key(key1_type&& k1) : k1_(std::move(k1)), k2_(), k_(1) { }

      /// Key2 constructor
      Key(key2_type&& k2) : k1_(), k2_(std::move(k2)), k_(2) { }

      /// Move constructor
      Key(Key_&& other) : k1_(std::move(other.k1_)), k2_(std::move(other.k2_)), k_(other.k_) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Destructor
      ~Key() { }

      /// Key assignment operator
      Key_& operator=(const Key_& other) {
        k1_ = other.k1_;
        k2_ = other.k2_;
        k_ = other.k_;

        return *this;
      }

      /// Assign Key1, Key2 is unassigned.
      Key_& operator=(const key1_type& k1) {
        k1_ = k1;
        k_ = 1;

        return *this;
      }

      /// Assign Key2, Key1 is unassigned.
      Key_& operator=(const key2_type& k2) {
        k2_ = k2;
        k_ = 2;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Key move assignment operator
      Key_& operator=(Key_&& other) {
        k1_ = std::move(other.k1_);
        k2_ = std::move(other.k2_);
        k_ = other.k_;

        return *this;
      }

      /// Assign Key1, Key2 is unassigned.
      Key_& operator=(key1_type&& k1) {
        k1_ = std::move(k1);
        k_ = 1;

        return *this;
      }

      /// Assign Key2, Key1 is unassigned.
      Key_& operator=(key2_type&& k2) {
        k2_ = std::move(k2);
        k_ = 2;

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Implicit key 1 conversion
      operator const key1_type() const {
        TA_ASSERT(k_ & 1, std::runtime_error, "Key1 is not set.");
        return k1_;
      }

      /// Implicit key 2 conversion
      operator const key2_type() const {
        TA_ASSERT(k_ & 2, std::runtime_error, "Key2 is not set.");
        return k2_;
      }

      /// Returns a constant reference to key 1.
      const key1_type& key1() const {
        TA_ASSERT(k_ & 1, std::runtime_error, "Key1 is not set.");
        return k1_;
      }

      /// Returns a constant reference to key 2.
      const key2_type& key2() const {
        TA_ASSERT(k_ & 2, std::runtime_error, "Key2 is not set.");
        return k2_;
      }

      /// Returns a flag with the keys that are assigned.
      unsigned int keys() const { return k_; }

      /// Set the keys to a new value.
      Key_ set(const key1_type& k1, const key2_type& k2) {
        k1_ = k1;
        k2_ = k2;
        k_ = 3;

        return *this;
      }

      /// Set Key1 to a new value, Key2 is unset.
      Key_ set(const key1_type& k1) {
        k1_ = k1;
        k_ = 1;

        return *this;
      }

      /// Set Key2 to a new value, Key1 is unset.
      Key_ set(const key2_type& k2) {
        k2_ = k2;
        k_ = 2;

        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Set the keys to a new value.
      Key_ set(key1_type&& k1, const key2_type& k2) {
        k1_ = std::move(k1);
        k2_ = k2;
        k_ = 3;

        return *this;
      }

      /// Set the keys to a new value.
      Key_ set(const key1_type& k1, key2_type&& k2) {
        k1_ = k1;
        k2_ = std::move(k2);
        k_ = 3;

        return *this;
      }

      /// Set the keys to a new value.
      Key_ set(key1_type&& k1, key2_type&& k2) {
        k1_ = std::move(k1);
        k2_ = std::move(k2);
        k_ = 3;

        return *this;
      }

      /// Set Key1 to a new value, Key2 is unset.
      Key_ set(key1_type&& k1) {
        k1_ = std::move(k1);
        k_ = 1;

        return *this;
      }

      /// Set Key2 to a new value, Key1 is unset.
      Key_ set(key2_type&& k2) {
        k2_ = std::move(k2);
        k_ = 2;

        return *this;
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    private:
      key1_type k1_;      ///< Key 1
      key2_type k2_;      ///< Key 2
      unsigned int k_;    ///< Flags for keys that are assigned.
    }; // class Key

    /// Compare two keys for equality (only compares key1).
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() == r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() == r.key2()));
    }

    /// Compare the key with a key1 type for equality.
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() == r;
    }

    /// Compare the key with a key1 type for equality.
    template<typename Key1, typename Key2>
    bool operator ==(const Key1& l, const Key<Key1, Key2>& r) {
      return l == r.key1();
    }

    /// Compare the key with a key2 type for equality.
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() == r;
    }

    /// Compare the key with a key2 type for equality.
    template<typename Key1, typename Key2>
    bool operator ==(const Key2& l, const Key<Key1, Key2>& r) {
      return l == r.key2();
    }


    /// Compare two keys for inequality (only compares key1).
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() != r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() != r.key2()));
    }

    /// Compare the key with a key1 type for inequality.
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() != r;
    }

    /// Compare the key with a key1 type for inequality.
    template<typename Key1, typename Key2>
    bool operator !=(const Key1& l, const Key<Key1, Key2>& r) {
      return l != r.key1();
    }

    /// Compare the key with a key2 type for inequality.
    template<typename Key1, typename Key2>
    bool operator !=(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() != r;
    }

    /// Compare the key with a key2 type for inequality.
    template<typename Key1, typename Key2>
    bool operator !=(const Key2& l, const Key<Key1, Key2>& r) {
      return l != r.key2();
    }

    /// Less-than comparison of two keys (only compares key1).
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() < r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() < r.key2()));
    }

    /// Less-than comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() < r;
    }

    /// Less-than comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator <(const Key1& l, const Key<Key1, Key2>& r) {
      return l < r.key1();
    }

    /// Less-than comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator <(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() < r;
    }

    /// Less-than comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator <(const Key2& l, const Key<Key1, Key2>& r) {
      return l < r.key2();
    }

    /// Less-than or equal-to comparison with two keys (only compares key1).
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() <= r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() <= r.key2()));
    }

    /// Less-than or equal-to comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() <= r;
    }

    /// Less-than or equal-to comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator <=(const Key1& l, const Key<Key1, Key2>& r) {
      return l <= r.key1();
    }

    /// Less-than or equal-to comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator <=(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() <= r;
    }

    /// Less-than or equal-to comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator <=(const Key2& l, const Key<Key1, Key2>& r) {
      return l <= r.key2();
    }

    /// Greater-than comparison with two keys (only compares key1).
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() > r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() > r.key2()));
    }

    /// Greater-than comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() > r;
    }

    /// Greater-than comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator >(const Key1& l, const Key<Key1, Key2>& r) {
      return l > r.key1();
    }

    /// Greater-than comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator >(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() > r;
    }

    /// Greater-than comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator >(const Key2& l, const Key<Key1, Key2>& r) {
      return l > r.key2();
    }

    /// Greater-than or equal-to comparison with two keys (only compares key1).
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return (((l.keys() & 1) && (r.keys() & 1)) && (l.key1() >= r.key1())) ||
          (((l.keys() & 2) && (r.keys() & 2)) && (l.key2() >= r.key2()));
    }

    /// Greater-than or equal-to comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>& l, const Key1& r) {
      return l.key1() >= r;
    }

    /// Greater-than or equal-to comparison of a key with a key1 type.
    template<typename Key1, typename Key2>
    bool operator >=(const Key1& l, const Key<Key1, Key2>& r) {
      return l >= r.key1();
    }

    /// Greater-than or equal-to comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator >=(const Key<Key1, Key2>& l, const Key2& r) {
      return l.key2() >= r;
    }

    /// Greater-than or equal-to comparison of a key with a key2 type.
    template<typename Key1, typename Key2>
    bool operator >=(const Key2& l, const Key<Key1, Key2>& r) {
      return l >= r.key2();
    }

  } // namespace detal
} // namespace TiledArray

#endif // TILEDARRAY_KEY_H__INCLUDED
