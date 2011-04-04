#ifndef TILEDARRAY_PACKED_TILE_H__INCLUDED
#define TILEDARRAY_PACKED_TILE_H__INCLUDED

#include <TiledArray/range.h>
#include <TiledArray/array_util.h>
#include <TiledArray/type_traits.h>
#include <boost/mpl/if.hpp>
#include <numeric>

namespace TiledArray {

  /// Cartesian packed tile

  /// PackedTile is a Cartesian packed tile reference to another, n-dimensional,
  /// tile. Packed tile does not contain element data. Instead, it forwards all
  /// function calls to the original tile when accessing the element data.
  /// Modifying packed tile data will modify the original tile's data. All other
  /// functions that reference tile dimensions are translated to the packed tile
  /// dimensions.
  template<typename T, unsigned int DIM>
  class PackedTile {
  public:
    typedef PackedTile<T, DIM> PackedTile_;
    typedef typename std::remove_const<T>::type tile_type;
    typedef typename tile_type::value_type value_type;
    typedef CoordinateSystem<DIM, tile_type::coordinate_system::dimension_order> coordinate_system;
    typedef typename tile_type::ordinal_type ordinal_type;
    typedef Range<ordinal_type, DIM, LevelTag<0>, coordinate_system > range_type;
    typedef typename range_type::index_type index_type;
    typedef typename range_type::size_array size_array;
    typedef typename range_type::volume_type volume_type;
    typedef typename range_type::const_iterator index_iterator;
    typedef typename tile_type::const_iterator const_iterator;
    typedef typename tile_type::iterator iterator;
    typedef typename boost::mpl::if_<std::is_const<T>, const value_type&, value_type&>::type reference_type;
    typedef const value_type & const_reference_type;

    static unsigned int dim() { return DIM; }

    /// Primary constructor.

    /// The packed tile is constructed with an existing tile and dimension
    /// boundaries. The boundaries are given with input iterators containing the
    /// boundaries. The boundaries must be in the form { b0, b1, ..., bn }, where
    /// 0 == b0 < b1 < ... < bn, n == DIM + 1, and DIM == the packed tile
    /// dimension.
    ///
    /// \arg \c t is the existing tile to be packed.
    /// \arg \c first, \c last are the dimension boundaries list.
    /// \arg \c origin is the offset of the packed tile (optional).
    template<typename InIter>
    PackedTile(T& t, InIter first, InIter last, const index_type& origin = index_type()) : r_(), w_(), t_(t)
    {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      TA_ASSERT( (valid_pack_(first, last, tile_type::dim, DIM)) , std::runtime_error,
          "Invalid packing information.");
      size_array s;
      s.assign(1);

      InIter next = first;
      typename size_array::iterator it = s.begin();
      for(++next; next != last; ++first, ++next, ++it ) {
        for(unsigned int d = *first; d < *next; ++d) {
          *it *= t.size()[d];
        }
      }

      TA_ASSERT( (std::accumulate(s.begin(), s.end(), ordinal_type(1), std::multiplies<ordinal_type>()) == t.volume()),
          std::runtime_error, "Packing list does not include the all dimensions of the original tile.");

      r_.resize(origin, origin + s);
      w_ = calc_weight_(r_.size());
    }

    /// Copy constructor
    PackedTile(const PackedTile_& other) : r_(other.r_), w_(other.w_), t_(other.t_) { }

    ~PackedTile() { }

    /// Assignment operator
    PackedTile_& operator =(const PackedTile_& other) {
      r_ = other.r_;
      w_ = other.w_;
      t_ = other.t_;

      return *this;
    }

    /// Returns an iterator to the first element of the tile.
    iterator begin() { return t_.begin(); } // no throw
    /// Returns a const_iterator to the first element of the tile.
    const_iterator begin() const { return t_.begin(); } // no throw
    /// Returns an iterator that points to the end of the tile.
    iterator end() { return t_.end(); } // no throw
    /// Returns a const_iterator that points to the end of the tile.
    const_iterator end() const { return t_.end(); } // no throw

    /// return a constant reference to the tile \c Range<> object.
    const range_type& range() const { return r_; } // no throw
    /// Returns the tile range start.
    const index_type& start() const { return r_.start(); } // no throw
    /// Returns the tile range finish.
    const index_type& finish() const { return r_.finish(); } // no throw
    /// Returns the tile range size.
    const size_array& size() const { return r_.size(); } // no throw
    /// Returns the number of elements in the volume.
    const volume_type volume() const { return r_.volume(); } // no throw
    /// Returns the precomputed dimension weights.
    const size_array& weight() const { return w_; } // no throw

    /// Returns true when index \c i is included in the tile.
    /// \arg \c i Element index.
    bool includes(const index_type& i) const { return r_.includes(i); }
    /// Returns true if index \c i is included in the tile.

    // The at() functions do error checking, but we do not need to implement it
    // here because the data container already does that. There is no need to do
    // it twice.
    /// Element access with range checking
    reference_type at(const ordinal_type& i) { return t_.at(i); }
    /// Element access with range checking
    const_reference_type at(const ordinal_type& i) const { return t_.at(i); }
    /// Element access with range checking
    reference_type at(const index_type& i){ return t_.at(ord_(i)); }
    /// Element access with range checking
    const_reference_type at(const index_type& i) const { return t_.at(ord_(i)); }

    /// Element access without error checking
    reference_type operator [](const ordinal_type& i) { return t_[i]; }
    /// Element access without error checking
    const_reference_type operator [](const ordinal_type& i) const { return t_[i]; }
    /// Element access without error checking
    reference_type operator [](const index_type& i) { return t_[ord_(i)]; }
    /// Element access without error checking
    const_reference_type operator [](const index_type& i) const { return t_[ord_(i)]; }

    /// Move the origin of the tile to the given index.

    /// The overall size and data of the tile are unaffected by this operation.
    /// \arg \c origin is the new lower bound for the tile.
    void set_origin(const index_type& origin) {
      r_.resize(origin, origin + r_.size());
    }

    /// Exchange calling tile's data with that of \c other.
    void swap(PackedTile_& other) {
      r_.swap(other.r_);
      boost::swap(w_, other.w_);
      std::swap(t_, other.t_);
    }

  private:

    /// Default construction not allowed.
    PackedTile();

    ordinal_type ord_(const index_type& i) const { // no throw
      return detail::calc_ordinal(i.data().begin(), i.data().end(), w_.begin());
    }

    /// Returns true if all dimension packing information follows the this
    /// pattern: a0 = 0, an = tile_dim, and a0 < a1 < a2 < ... < an.
    template<typename InIter>
    static bool valid_pack_(InIter first, InIter last, const unsigned int tile_dim, const unsigned int pack_dim) {
      if(first == last && tile_dim != 0)
        return false;

      if(*first != 0)
        return false;

      InIter next = first;
      ++next;
      if(next == last)
        return false;
      unsigned int d = 0;
      for(; next != last; ++next, ++first, ++d)
        if(*first >= *next)
          return false;

      if(*first != tile_dim)
        return false;
      if(d != pack_dim)
        return false;

      return true;
    }

    /// Class wrapper function for detail::calc_weight() function.
    static size_array calc_weight_(const size_array& size) { // no throw
      size_array result;
      detail::calc_weight(coordinate_system::begin(size), coordinate_system::end(size),
          coordinate_system::begin(result));
      return result;
    }

    range_type r_;   ///< Packed tile range information
    size_array w_;   ///< Range weights
    T& t_;   ///< Reference to tile
  };

} // namespace TiledArray

#endif // TILEDARRAY_PACKED_TILE_H__INCLUDED
