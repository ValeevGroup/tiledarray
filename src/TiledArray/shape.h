/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/bitset.h>
#include <TiledArray/madness.h>
#include <functional>

namespace TiledArray {
  namespace detail {

    /// Shape is used to store an estimate of the magnitude of a tile

    /// This object stores a scalar that represents the magnitude of a tile. It
    /// is is used to gauge weather a tile should be stored and/or included in
    /// a calculation. A tile is assumed to be zero when the stored scalar value
    /// is below the given threshold.
    /// \tparam T The type used to represent the tile magnitude
    template <typename T, typename Comp = std::less<T> >
    class Shape {
    public:
      typedef Shape<T, Comp> Shape_;
      typedef Tensor<T> tensor_type; ///< Tensor representation type
      typedef typename tensor_type::value_type value_type; ///< The type used to represent the tile magnitude
      typedef typename tensor_type::range_type range_type; ///< The shape range type
      typedef typename tensor_type::reference reference; ///< Tile estimate reference type
      typedef typename tensor_type::const_reference const_reference; ///< Tile estimate reference type

      typedef Comp comp_type; ///< Comparison function type

    private:
      Tensor<value_type> data_; ///< Tile magnitude estimate
      value_type threshold_; ///< Zero threshold for a tile
      bool shared_; ///< true if data is distributed, false if it is local
      comp_type comp_; ///< Comparison function

      template <typename, typename> friend class Shape;

    public:
      /// Construct an empty shape
      Shape() : data_(), threshold_(), shared_(true), comp_() { }

      /// Shape copy constructor

      /// \param other The shape object to be copied
      Shape(const Shape_& other) :
        data_(other.data_), threshold_(other.threshold_), shared_(other.shared_),
        comp_(other.comp_)
      { }

      /// Construct a shape that describes a shape over \c range

      /// \param range The range that shape will describe
      /// \param threshold The zero threshold for tiles
      /// \param init_value The starting value for elements in the shape
      Shape(const range_type& range, const value_type threshold,
          const value_type& init_value = value_type(),
          const comp_type& comp = comp_type()) :
        data_(range, init_value), threshold_(threshold), shared_(false), comp_(comp)
      { }

      /// Construct shape with initial data for a set of tiles

      /// Construct a shape with a set of tile estimates specified by an iterator
      /// range. \c first must dereference to a \c std::pair<Index, value_type> ,
      /// where \c Index is an coordinate array type or an integral type. The
      /// index of the pair must be contained within \c range .
      /// \param range The range that shape will describe
      /// \param threshold The zero threshold for tiles
      /// \param first An iterator to the first element of tile estimate data
      /// \param last An iterator to the last element of tile estimate data
      template <typename InIter>
      Shape(const range_type& range, const value_type threshold, InIter first,
          InIter last, const comp_type& comp = comp_type()) :
        data_(range), threshold_(threshold), shared_(false), comp_()
      {
        // Initialize the local tile zero estimates
        for(; first != last; ++first)
          data_[first->first] = first->second;
      }

      /// Copy assignment operator

      /// \param other The shape to be copied
      /// \return A reference to this object
      Shape_& operator=(const Shape_& other) {
        data_ = other.data_;
        threshold_ = other.threshold_;
        shared_ = other.shared_;
        comp_ = other.comp_;

        return *this;
      }

      /// Assign data from tensor
      template <typename U, typename A>
      Shape_& operator=(const Tensor<U, A>& tensor) {
        data_ = tensor;
        return *this;
      }

      template <typename S, typename U, typename CU, typename V, typename CV>
      Shape_& add_and_scale(const S scalar, const Shape<U, CU>& left, const Shape<V, CV>& right) {
        TA_ASSERT(left.shared_);
        TA_ASSERT(right.shared_);

        if(left.range().volume() != 0ul) {
          if(right.range().size() != 0ul) {
            TA_ASSERT(left.range() == right.range());
            if(data_.empty()) {
              data_ = tensor_type(left.range(), scalar, left.begin(),
                  right.begin(), std::plus<value_type>());
            } else {
              for(std::size_t i = 0; i < data_.size(); ++i)
                data_[i] = scalar * (left.data_[i] + right.data_[i]);
            }
            threshold_ = left.threshold_ + right.threshold_;
          } else {
            for(std::size_t i = 0; i < data_.size(); ++i)
              data_[i] = scalar * left.data_[i];
          }
        } else {
          if(right.range().size() != 0ul) {
            TA_ASSERT(data_.range() == right.range());
            for(std::size_t i = 0; i < data_.size(); ++i)
              data_[i] = scalar * right.data_[i];
          } else {
            data_ = tensor_type();
          }
        }

        shared_ = true;

        return *this;
      }


      /// Tensor accessor

      /// Convert the shape into a tensor object where each element of the
      /// tensor represents the tile estimate value
      /// \return A tensor that represents the shape
      const tensor_type& tensor() const {
        return data_;
      }

      /// Tensor accessor

      /// Convert the shape into a tensor object where each element of the
      /// tensor represents the tile estimate value
      /// \return A tensor that represents the shape
      tensor_type& tensor() {
        return data_;
      }

      /// Shape range data accessor

      /// \return A const reference to the shape range
      const range_type& range() const { return data_.range(); }

      /// Share local data among all processes in \c world

      /// The function will perform an "all reduce" reduce operation to share
      /// local data among all nodes in world.
      /// \tparam Op The reduction operation type
      /// \param world The world that will share data
      /// \param op The reduction operation used to share data among nodes
      /// \note This operation must be called from the main thread.
      template <typename Op>
      void share(madness::World& world, const Op& op) {
        TA_ASSERT(! shared_);
        if(! data_.empty())
          world.gop.reduce(data_.data(), data_.range().volume(), op);
        shared_ = true;
      }

      /// Share local data among all processes in \c world

      /// The function will perform an "all reduce" reduce operation to share
      /// local data among all nodes in world.
      /// \param world The world that will share data
      /// \param op The reduction operation used to share data
      /// \note This operation must be called from the main thread.
      void share(madness::World& world) {
        TA_ASSERT(! shared_);
        if(! data_.empty())
          world.gop.sum(data_.data(), data_.range().volume());
        shared_ = true;
      }

      /// Shared accessor

      /// \return \c true if the data has been shared with all processes.
      bool is_shared() const { return shared_; }

      /// Disable data sharing

      /// Allows shape to be used without sharing data among nodes.
      void no_share() { shared_ = true; }

      /// Threshold value

      /// \return The current zero threshold
      value_type threshold() const { return threshold_; }

      /// Set a new threshold value

      /// \param new_threshold The new threshold value
      void threshold(const value_type new_threshold) { threshold_ = new_threshold; }

      /// Comparison function accessor

      /// \return A const reference to the comparison function
      const comp_type& comp() { return comp_; }

      /// Set tile estimate value

      /// \tparam Index The tile index type
      /// \param i The index of the tile
      /// \param value The tile estimate value
      /// \throw When the data has already been shared
      template <typename Index>
      void set(const Index& i, const_reference value) {
        TA_ASSERT(! shared_);
        TA_ASSERT(data_.range().includes(i));
        data_[i] = value;
      }


      /// Check for a dense shape

      /// \return True if is_zero() evaluates to true for all tiles.
      bool is_dense() const {
        TA_ASSERT(shared_);
        if(! data_.empty())
          for(typename Tensor<value_type>::const_iterator it = data_.begin(); it != data_.end(); ++it)
            if(comp_(*it, threshold_))
              return false;

        return true;
      }

      /// Check for a zero estimate

      /// \return \c true when the comparison operation between the tile estimate
      /// and the threshold returns \c true .
      template <typename Index>
      bool is_zero(const Index& i) const {
        TA_ASSERT(shared_);
        if(data_.empty())
          return false;
        TA_ASSERT(data_.range().includes(i));
        return comp_(data_[i], threshold_);
      }

      /// Exchange the data of this Shape with \c other Shape

      /// \param other The shape to exchage data with
      void swap(Shape_& other) {
        data_.swap(other.data_);
        std::swap(threshold_, other.threshold_);
        std::swap(shared_, other.shared_);
        std::swap(comp_, other.comp_);
      }

    }; // class Shape

    template <>
    class Shape<bool, std::equal_to<bool> > {
    public:
      typedef Shape<bool, std::equal_to<bool> > Shape_;
      typedef bool value_type; ///< The type used to represent the tile magnitude
      typedef Tensor<unsigned int> tensor_type;
      typedef tensor_type::range_type range_type;
      typedef TiledArray::detail::Bitset<>::reference reference; ///< Tile estimate reference type
      typedef TiledArray::detail::Bitset<>::const_reference const_reference; ///< Tile estimate reference type
      typedef std::equal_to<bool>  comp_type; ///< Comparison function type

    private:
      range_type range_; ///< Shape range
      TiledArray::detail::Bitset<> data_; ///< Tile magnitude estimate
      bool shared_; ///< true if data is distributed, false if it is local

    public:
      /// Construct an empty shape
      Shape() : range_(), data_(0), shared_(false) { }

      /// Shape copy constructor

      /// \param other The shape object to be copied
      Shape(const Shape_& other) :
        range_(other.range_), data_(other.data_), shared_(other.shared_)
      { }

      /// Construct a shape that describes a shape over \c range

      /// \param range The range that shape will describe
      /// \param threshold The zero threshold for tiles
      /// \param init_value The starting value for elements in the shape
      Shape(const range_type& range, const value_type threshold,
          const value_type& init_value = value_type()) :
        range_(range), data_(range_.volume()), shared_(false)
      {
        if(init_value)
          data_.set();
      }

      /// Construct shape with initial data for a set of tiles

      /// Construct a shape with a set of tile estimates specified by an iterator
      /// range. \c first must dereference to a \c std::pair<Index, value_type> ,
      /// where \c Index is an coordinate array type or an integral type. The
      /// index of the pair must be contained within \c range .
      /// \param range The range that shape will describe
      /// \param threshold The zero threshold for tiles
      /// \param first An iterator to the first element of tile estimate data
      /// \param last An iterator to the last element of tile estimate data
      template <typename InIter>
      Shape(const range_type& range, const value_type threshold, InIter first,
          InIter last) :
        range_(range), data_(range_.volume()), shared_(false)
      {
        // Initialize the local tile zero estimates
        for(; first != last; ++first) {
          TA_ASSERT(range_.includes(first->first));
          data_.set(range_.ord(first->first), first->second);
        }
      }

      /// Assignment operator

      /// \param other The shape to be copied
      /// \return A reference to this object
      Shape_& operator=(const Shape_& other) {
        // Copy shape data
        range_ = other.range_;
        data_ = other.data_;
        shared_ = other.shared_;

        return *this;
      }

      /// Tensor conversion operator

      /// Convert the shape into a tensor object where each element of the
      /// tensor represents the tile estimate value
      /// \return A tensor that represents the shape
      operator tensor_type () const {
        // Create a temporary tensor object
        tensor_type temp(range_, 0u);

        // Fill the tensor with data_
        for(std::size_t i = 0ul; i < temp.size(); ++i)
          if(data_[i])
            temp[i] = 1u;

        return temp;
      }

      /// Shape range data accessor

      /// \return A const reference to the shape range
      const range_type& range() const { return range_; }

      /// Share local data among all processes in \c world

      /// The function will perform an "all reduce" reduce operation to share
      /// local data among all nodes in world.
      /// \tparam Op The reduction operation type
      /// \param world The world that will share data
      /// \param op The reduction operation used to share data among nodes
      /// \note This operation must be called from the main thread.
      template <typename Op>
      void share(madness::World& world, const Op& op) {
        if(! shared_) {
          world.gop.reduce(data_.get(), data_.num_blocks(), op);
          shared_ = true;
        }
      }

      /// Share local data among all processes in \c world

      /// The function will perform an "all reduce" reduce operation to share
      /// local data among all nodes in world.
      /// \param world The world that will share data
      /// \param op The reduction operation used to share data
      /// \note This operation must be called from the main thread.
      template <typename Op>
      void share(madness::World& world) {
        if(! shared_) {
          world.gop.bit_or(data_.get(), data_.num_blocks());
          shared_ = true;
        }
      }

      /// Disable data sharing among nodes

      /// Allows shape to be used without sharing data among nodes.
      void no_share() {
        TA_ASSERT(! shared_);
        shared_ = true;
      }

      /// Set tile estimate value

      /// \tparam Index The tile index type
      /// \param i The index of the tile
      /// \param value The tile estimate value
      template <typename Index>
      void set(const Index& i, const_reference value) {
        data_.set(i, value);
      }

      /// Check for a dense shape

      /// \return True if is_zero() evaluates to true for all tiles.
      bool is_dense() const {
        return data_.count() == data_.size();
      }

      /// Zero test for tile

      /// \tparam Index The index type (integral or coordinate array type)
      /// \param i The index of the tile to test
      /// \return \c true when the comparison of data associated with tile \c i
      /// is true relative to the threshold
      template <typename Index>
      bool is_zero(const Index& i) const {
        TA_ASSERT(shared_);
        TA_ASSERT(range_.includes(i));
        return data_[i];
      }

      /// Exchange the data of this Shape with \c other Shape

      /// \param other The shape to exchage data with
      void swap(Shape_& other) {
        range_.swap(other.range_);
        data_.swap(other.data_);
        std::swap(shared_, other.shared_);
      }

    }; // class Shape<bool, std::equal_to<bool> >

    typedef Shape<bool, std::equal_to<bool> > ShapeBool;

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
