#ifndef TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SPARSE_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/shape.h>
#include <TiledArray/utility.h>
#include <TiledArray/bitset.h>

namespace TiledArray {

  template<typename CS>
  class SparseShape : public Shape<CS>
  {
  protected:
    typedef SparseShape<CS> SparseShape_;
    typedef Shape<CS> Shape_;
    typedef madness::WorldObject<SparseShape_ > WorldObject_;

  private:
    typedef unsigned long block_type;

  public:
    typedef CS coordinate_system;                         ///< Shape coordinate system
    typedef typename Shape_::index index;                 ///< index type
    typedef typename Shape_::ordinal_index ordinal_index; ///< ordinal index type
    typedef typename Shape_::range_type range_type;       ///< Range type
    typedef typename Shape_::pmap_type pmap_type;         ///< Process map type
    typedef typename Shape_::array_type array_type;       ///< Dense array type

  private:
    // not allowed
    SparseShape();
    SparseShape_& operator=(const SparseShape&);

    /// Copy constructor
    SparseShape(const SparseShape_& other) :
      Shape_(other),
      world_(other.world_),
      tiles_(other.tiles_)
    { }

  public:
    /// Primary constructor

    /// \tparam InIter Input list input iterator type
    /// \param w The world where this shape lives
    /// \param r The range object associated with this shape
    /// \param m The process map for this shape
    /// \param first First element of a list of tiles that will be stored locally
    /// \param last Last element of a list of tiles that will be stored locally
    /// \note Tiles in the list that are not owned by this process (according to
    /// the process map) are ignored.
    template <typename InIter>
    SparseShape(const madness::World& w, const range_type& r, const pmap_type& m,
        InIter first, InIter last) :
      Shape_(r,m),
      world_(w),
      tiles_(r.volume())
    {

      ordinal_index o = 0;
//      const int rank = world_.rank();
      for(; first != last; ++first) {
//        if(Shape_::owner(*first) == rank) {
          o = Shape_::ord(*first);
          tiles_.set(o);
//        }
      }

      // Construct the bitset for remote data

      world_.gop.bit_or(tiles_.get(), tiles_.num_blocks());
    }

    /// Map constructor constructor

    /// \param w The world where this shape lives
    /// \param r The range object associated with this shape
    /// \param m The process map for this shape
    /// \param first First element of a list of tiles that will be stored locally
    /// \param last Last element of a list of tiles that will be stored locally
    /// \note Tiles in the list that are not owned by this process (according to
    /// the process map) are ignored.
    SparseShape(const madness::World& w, const range_type& r, const pmap_type& m,
        const array_type a) :
      Shape_(r,m),
      world_(w),
      tiles_(r.volume())
    {

      ordinal_index o = 0;
      for(typename array_type::const_iterator it = a.begin(); it != a.end(); ++it, ++o) {
        if(*it != 0)
          tiles_.set(o);
      }
    }

    virtual ~SparseShape() { }

    /// Construct a copy of this shape

    /// \return A shared pointer to a copy of this object
    virtual std::shared_ptr<Shape_> clone() const {
      return std::shared_ptr<Shape_>(static_cast<Shape_*>(new SparseShape_(*this)));
    }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const { return typeid(SparseShape_); }

    /// Construct a shape map

    /// \return A dense array that contains 1 where tiles exist in the shape and
    /// 0 where tiles do not exist in the shape.
    virtual array_type make_shape_map() const {
      array_type result(this->range(), 0);
      std::size_t vol = this->range().volume();
      for(std::size_t i = 0; i < vol; ++i)
        if(tiles_[i])
          result[i] = 1;
      return result;
    }

  private:

    /// Probe for the presence of a tile in the shape

    /// \param k The index to be probed.
    virtual bool local_probe(const ordinal_index& i) const {
      return tiles_[i];
    }

    const madness::World& world_;
    detail::Bitset<> tiles_;
  }; // class SparseShape

} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
