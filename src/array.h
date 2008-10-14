#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <cassert>
#include <boost/shared_ptr.hpp>

#include <tuple.h>
#include <shape.h>
#include <policy.h>
#include <madness_runtime.h>

namespace TiledArray {
    
  template <typename T, unsigned int DIM, typename ArrayPolicy = DefaultArrayPolicy<T,DIM> > class Array :
      public DistributedObject< Array<T,DIM,ArrayPolicy> > {
  public:
    typedef DistributedObject< Array<T,DIM,ArrayPolicy> > DistributedObjectBase;
    typedef ArrayPolicy Policy;
    typedef Tuple<DIM> TileIndex;
    typedef Tuple<DIM> ElementIndex;
    typedef typename Policy:: template Hash<TileIndex> IndexToKey;
    typedef typename IndexToKey::Key TileKey;
    typedef typename Policy::Tile Tile;
    
    typedef DistributedContainer< TileKey, Tile > TileContainer;
    typedef typename TileContainer::futureT TileResult;

    /// array is defined by its shape
    Array(DistributedRuntime& rtime, const boost::shared_ptr<AbstractShape<DIM> >& shp) :
      DistributedObjectBase(rtime),
      // deep copy of shape
      shape_(shp->clone()), data_()
    {
    }
    /// assign each element to a
    void assign(T a);
    
    /// where is tile k
    DistributedProcessID owner(const TileIndex& k) {
      return data_.owner( IndexToKey(k) );
    }
    /// access tile k
    TileResult tile(const TileIndex& k) {
      return data_.find( IndexToKey(k) );
    }

    /// make new Array by applying permutation P
    Array transpose(const Tuple<DIM>& P);

    /// bind a string to this array to make operations look normal
    /// e.g. R("ijcd") += T2("ijab") . V("abcd") or T2new("iajb") = T2("ijab")
    /// runtime then can figure out how to implement operations
    //ArrayExpression operator()(const char*) const;
    
  private:
    boost::shared_ptr< AbstractShape<DIM> > shape_;
    TileContainer data_;
  };
  
};

#endif // TILEDARRAY_H__INCLUDED
