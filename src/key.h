#ifndef TILEDARRAY_KEY_H__INCLUDED
#define TILEDARRAY_KEY_H__INCLUDED

namespace TiledArray {
  namespace detail {

    /// Key class that holds two arbitrary key types.

    /// Contains two arbitrary key values. It provides methods of comparing the
    /// two keys with the appropriate type. Both key values must be set.
    template<typename Key1, typename Key2>
    class Key {
    public:
      typedef Key<Key1, Key2> Key_;
      typedef Key1 key2_type;
      typedef Key2 key1_type;

      /// Default constructor
      Key() : k1_(), k2_() { }

      /// Key pair constructor
      Key(const key1_type& k1, const key2_type& k2) : k1_(k1), k2_(k2) { }

      /// Copy constructor
      Key(const Key_& other) : k1_(other.k1_), k2_(other.k2) { }
#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Move constructor for key 2
      Key(const key1_type& k1, key2_type&& k2) : k1_(k1), k2_(std::move(k2)) { }
      /// Move constructor for key 1
      Key(key1_type&& k1, const key2_type& k2) : k1_(std::move(k1)), k2_(k2) { }
      /// Move constructor for key 1 and key 2
      Key(key1_type&& k1, key2_type&& k2) : k1_(std::move(k1)), k2_(std::move(k2)) { }
      /// Move constructor
      Key(Key_&&) : k_(other.k_), k1_(std::move(other.k1_)), k2_(std::move(other.k2_)) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Destructor
      ~Key() { }

      /// Key assignment operator
      Key_& operator=(const Key_& other) {
        k1_ = other.k1_;
        k2_ = other.k2_;
        return *this;
      }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
      /// Key move assignment operator
      Key_& operator=(Key_&& other) {
        k_ = other.k_;
        k1_ = std::move(other.k1_);
        k2_ = std::move(other.k2_);
      }
#endif // __GXX_EXPERIMENTAL_CXX0X__

      /// Implicit key 1 conversion
      operator key1_type() const { return k1_; }

      /// Implicit key 2 conversion
      operator key2_type() const { return k2_; }

      /// Returns a constant reference to key 1.
      const key1_type& key1() const { return k1_; }

      /// Returns a constant reference to key 2.
      const key2_type& key2() const { return k2_; }

      /// Set the keys to a new value.
      void set(const key1_type& k1, const key2_type& k2) {
        k1_ = k1;
        k2_ = k2;
      }

    private:
      key1_type k1_;      ///< Key 1
      key2_type k2_;      ///< Key 2
    }; // class Key

    /// Compare two keys for equality (only compares key1).
    template<typename Key1, typename Key2>
    bool operator ==(const Key<Key1, Key2>& l, const Key<Key1, Key2>& r) {
      return l.key1() == r.key1();
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
      return l.key1() != r.key1();
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
      return l.key1() < r.key1();
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
      return l.key1() <= r.key1();
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
      return l.key1() > r.key1();
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
      return l.key1() >= r.key1();
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
