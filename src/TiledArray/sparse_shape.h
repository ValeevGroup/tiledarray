#ifndef TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
#define TILEDARRAY_SPARSE_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/shape.h>
#include <TiledArray/utility.h>
#include <TiledArray/bitset.h>

namespace TiledArray {

  template<typename Container>
  class SparseShape : public Shape<Container>
  {
  protected:
    typedef SparseShape<Container> SparseShape_;
    typedef Shape<Container> Shape_;

  private:
    typedef unsigned long block_type;

  public:
    typedef typename Shape_::size_type size_type;

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
    SparseShape(const madness::World& w, const Container& c, InIter first, InIter last) :
      Shape_(c),
      world_(w),
      tiles_(c.volume())
    {
      for(; first != last; ++first)
        tiles_.set(Shape_::ord(*first));

      // Construct the bitset for remote data

      world_.gop.bit_or(tiles_.get(), tiles_.num_blocks());
    }

    virtual ~SparseShape() { }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const { return typeid(SparseShape_); }

  private:

    /// Probe for the presence of a tile in the shape

    /// \param k The index to be probed.
    virtual bool local_probe(const size_type& i) const {
      return tiles_[i];
    }

    const madness::World& world_;
    detail::Bitset<> tiles_;
  }; // class SparseShape

} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_SHAPE_H__INCLUDED
