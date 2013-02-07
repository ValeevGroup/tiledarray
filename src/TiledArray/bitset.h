/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_BITSET_H__INCLUDED
#define TILEDARRAY_BITSET_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/transform_iterator.h>
#include <climits>

namespace TiledArray {
  namespace detail {

    /// Fixed size bitset

    /// Bitset is similar to \c std::bitset except the size is set at runtime.
    /// This bit set has very limited functionality, because it is only intended
    /// to fit the needs of \c SparseShape. The block type may be an integral
    /// or char type.
    /// \tparam Block The type used to store the data [ default = \c unsigned \c long ]
    template <typename Block = unsigned long>
    class Bitset {
    private:
      TA_STATIC_ASSERT( (std::is_integral<Block>::value || std::is_same<Block, char>::value) );

      static const std::size_t block_bits; ///< The number of bits in a block

    public:

      class reference {

        friend class Bitset<Block>;

        reference(Block& block, Block mask) : block_(block), mask_(mask) { }

        // Not allowed
        void operator&();

      public:

        reference& operator=(bool value) {
          assign(value);
          return *this;
        }

        reference& operator=(const reference& other) {
          assign(other);
          return *this;
        }

        reference& operator|=(bool value) {
          if(value) set();
          return *this;
        }

        reference& operator&=(bool value) {
          if(!value) reset();
          return *this;
        }

        reference& operator^=(bool value) {
          if(value) flip();
          return *this;
        }

        reference& operator-=(bool value) {
          if(value) reset();
          return *this;
        }

        operator bool() const { return block_ & mask_; }

        bool operator~() const { return !(block_ & mask_); }

        reference& flip() {
          block_ ^= mask_;
          return *this;
        }

     private:

        void assign(bool value) {
          if(value)
            set();
          else
            reset();
        }

        void set() { block_ |= mask_; }

        void reset() { block_ &= ~mask_; }

        Block& block_;
        const Block mask_;
      }; // class reference

    private:

      class ConstTransformOp {
      public:
        typedef std::size_t argument_type;
        typedef Block result_type;

        ConstTransformOp(const Bitset<Block>& bitset) : bitset_(&bitset) { }

        ConstTransformOp(const ConstTransformOp& other) : bitset_(other.bitset_) { }


        ConstTransformOp& operator=(const ConstTransformOp& other) {
          bitset_ = other.bitset_;

          return *this;
        }

        Block operator()(std::size_t i) const { return (*bitset_)[i]; }

      private:
        const Bitset<Block>* bitset_;
      }; // class ConstTransformOp

      class TransformOp {
      public:
        typedef std::size_t argument_type;
        typedef reference result_type;

        TransformOp(const Bitset<Block>& bitset) : bitset_(bitset) { }
        reference operator()(std::size_t i) const { return const_cast<Bitset<Block>&>(bitset_)[i]; }
      private:
        const Bitset<Block>& bitset_;
      }; // class TransformOp

    public:
      typedef Bitset<Block> Bitset_;
      typedef Block block_type;     ///< The type used to store the data
      typedef Block value_type;
      typedef Block const_reference; ///< Constant reference to a bit
      typedef std::size_t size_type;
      typedef UnaryTransformIterator<Block, ConstTransformOp> const_iterator;
      typedef UnaryTransformIterator<Block, TransformOp> iterator;

      /// Construct a bitset that contains \c s bits.

      /// \param s The number of bits
      /// \throw std::bad_alloc If bitset allocation fails.
      Bitset(size_type s) :
          size_(s),
          blocks_((size_ / block_bits) + (size_ % block_bits ? 1 : 0)),
          set_((size_ ? new block_type[blocks_] : NULL))
      {
        std::fill_n(set_, blocks_, block_type(0));
      }

      /// Construct a bitset that contains \c s bits.

      /// \param s The number of bits
      /// \throw std::bad_alloc If bitset allocation fails.
      template <typename InIter>
      Bitset(InIter first, InIter last) :
          size_(std::distance(first, last)),
          blocks_((size_ / block_bits) + (size_ % block_bits ? 1 : 0)),
          set_((size_ ? new block_type[blocks_] : NULL))
      {
        // Initialize to zero
        std::fill_n(set_, blocks_, block_type(0));

        for(size_type i = 0; first != last; ++i, ++first)
          if(*first)
            set(i);
      }

      /// Copy constructor for bitset

      /// \param other The bitset to copy
      /// \throw std::bad_alloc If bitset allocation fails.
      Bitset(const Bitset<Block>& other) :
          size_(other.size_),
          blocks_(other.blocks_),
          set_((size_ ? new block_type[blocks_] : NULL))
      {
        std::copy(other.set_, other.set_ + blocks_, set_);
      }

      ~Bitset() { delete [] set_; }

      /// Assignment operator

      /// This will only copy the data from \c other. It will not change the size
      /// of the bitset.
      /// \param other The bitset to copy
      /// \throw std::runtime_error If the bitset sizes are not equal.
      Bitset<Block>& operator=(const Bitset<Block>& other) {
        TA_ASSERT(size_ == other.size_);
        if(this != &other)
          std::copy(other.set_, other.set_ + blocks_, set_);

        return *this;
      }

      /// Or-assignment operator

      /// Or-assign all bits from the two ranges
      /// \param other The bitset to be or-assigned to this bitset
      /// \throw std::range_error If the bitset sizes are not equal.
      Bitset<Block>& operator|=(const Bitset<Block>& other) {
        TA_ASSERT(size_ == other.size_);
        for(size_type i = 0; i < blocks_; ++i)
          set_[i] |= other.set_[i];

        return *this;
      }

      /// And-assignment operator

      /// And-assign all bits from the two ranges
      /// \param other The bitset to be and-assigned to this bitset
      /// \throw std::range_error If the bitset sizes are not equal.
      Bitset<Block>& operator&=(const Bitset<Block>& other) {
        TA_ASSERT(size_ == other.size_);
        for(size_type i = 0; i < blocks_; ++i)
          set_[i] &= other.set_[i];

        return *this;
      }


      /// And-assignment operator

      /// And-assign all bits from the two ranges
      /// \param other The bitset to be and-assigned to this bitset
      /// \throw std::range_error If the bitset sizes are not equal.
      Bitset<Block>& operator^=(const Bitset<Block>& other) {
        TA_ASSERT(size_ == other.size_);
        for(size_type i = 0; i < blocks_; ++i)
          set_[i] ^= other.set_[i];

        return *this;
      }

      /// Bit accessor operator

      /// \param i The bit to access
      /// \return The value of i-th bit of the bit set
      /// \throw std::out_of_range If \c i is greater than or equal to the size
      const_reference operator[](size_type i) const {
        TA_ASSERT(i < size_);
        return mask(i) & set_[block_index(i)];
      }

      reference operator[](size_type i) {
        TA_ASSERT(i < size_);
        return reference(set_[block_index(i)], mask(i));
      }

      const_iterator begin() const {
        return const_iterator(0, ConstTransformOp(*this));
      }

      iterator begin() {
        return iterator(0, TransformOp(*this));
      }

      const_iterator end() const {
        return const_iterator(size_, ConstTransformOp(*this));
      }

      iterator end() {
        return iterator(size_, TransformOp(*this));
      }

      /// Set a bit value

      /// \param i The bit to be set
      /// \param value The new value of the bit
      /// \throw std::out_of_range When \c i is >= size.
      void set(size_type i, bool value = true) {
        TA_ASSERT(i < size_);
        if(value)
          set_[block_index(i)] |= mask(i);
        else
          reset(i);
      }

      /// Set all bits

      /// \throw nothing
      void set() { std::fill_n(set_, blocks_, ~block_type(0)); }

      /// Reset a bit

      /// \param i The bit to be reset
      /// \throw std::out_of_range When \c i is >= size.
      void reset(size_type i) {
        TA_ASSERT(i < size_);
        set_[block_index(i)] &= ~mask(i);
      }

      /// Set all bits

      /// \throw nothing
      void reset() {
        std::fill_n(set_, blocks_, block_type(0));
      }

      /// Flip a bit

      /// \param i The bit to be flipped
      /// \throw std::out_of_range When \c i is >= size.
      void flip(size_type i) {
        TA_ASSERT(i < size_);
        set_[block_index(i)] ^= mask(i);
      }

      /// Flip all bits

      /// \throw nothing
      void flip() {
        for(size_type i = 0; i < blocks_; ++i)
          set_[i] = ~set_[i];
      }

      /// Count the number of non-zero bits

      /// \return The number of non-zero bits
      size_type count() const {
        size_type c = 0ul;
        for(size_type i = 0ul; i < blocks_; ++i) {
          block_type v = set_[i]; // temp
          v = v - ((v >> 1) & (block_type)~(block_type)0 / 3);
          v = (v & (block_type)~(block_type)0 / 15 * 3) + ((v >> 2) & (block_type)~(block_type)0 / 15 * 3);
          v = (v + (v >> 4)) & (block_type)~(block_type)0 / 255 * 15;
          c += (block_type)(v * ((block_type)~(block_type)0 / 255)) >> (sizeof(block_type) - 1) * CHAR_BIT; // count
        }
        return c;
      }

      /// Data pointer accessor

      /// The pointer to the data points to a contiguous block of memory of type
      /// \c block_type that contains \c num_blocks() elements.
      /// \return A pointer to the first element of the bitset data
      /// \throw nothing
      const block_type* get() const { return set_; }


      /// Data pointer accessor

      /// The pointer to the data points to a contiguous block of memory of type
      /// \c block_type that contains \c num_blocks() elements.
      /// \return A pointer to the first element of the bitset data
      /// \throw nothing
      block_type* get() { return set_; }

      /// Bitset size

      /// \return Number of bits in the bitset
      /// \throw nothing
      size_type size() const { return size_; }

      /// Bitset block size

      /// \return Number of block_type elements used to store the bitset array.
      /// \throw nothing
      size_type num_blocks() const { return blocks_; }

      void swap(Bitset_& other) {
        std::swap(size_, other.size_);
        std::swap(blocks_, other.blocks_);
        std::swap(set_, other.set_);
      }

    private:

      /// Calculate block index

      /// \return The block index that contains the i-th bit
      /// \throw nothing
      static size_type block_index(size_type i) { return i / block_bits; }

      /// Calculate the bit index

      /// \return The bit index that contains the i-th bit in the block
      /// \throw nothing
      static size_type bit_index(size_type i) { return i % block_bits; }

      /// Construct a mask

      /// \return A \c block_type that contains a single bit "on" bit at the i-th
      /// bit index.
      static block_type mask(size_type i) { return block_type(1) << bit_index(i); }

      size_type size_;    ///< The number of bits in the set
      size_type blocks_;  ///< The number of blocks used to store the bits
      block_type* set_;   ///< An array that store the bits
    }; // class Bitset

    template <typename B>
    inline void swap(Bitset<B>& b0, Bitset<B>& b1) {
      b0.swap(b1);
    }

    template <typename Block>
    const std::size_t Bitset<Block>::block_bits =
        8 * sizeof(typename Bitset<Block>::block_type);

    /// Bitwise and operator of bitset.

    /// \tparam Block The bitset block type
    /// \param left The left-hand bitset
    /// \param right The right-hand bitset
    /// \return The a intersection of the \c left and \c right bitsets
    template <typename Block>
    Bitset<Block> operator&(Bitset<Block> left, const Bitset<Block>& right) {
      left &= right;
      return left;
    }

    /// Bitwise or operator of bitset.

    /// \tparam Block The bitset block type
    /// \param left The left-hand bitset
    /// \param right The right-hand bitset
    /// \return The union of the \c left and \c right bitsets
    template <typename Block>
    Bitset<Block> operator|(Bitset<Block> left, const Bitset<Block>& right) {
      left |= right;
      return left;
    }


    /// Bitwise xor operator of bitset.

    /// \tparam Block The bitset block type
    /// \param left The left-hand bitset
    /// \param right The right-hand bitset
    /// \return The union of the \c left and \c right bitsets
    template <typename Block>
    Bitset<Block> operator^(Bitset<Block> left, const Bitset<Block>& right) {
      left ^= right;
      return left;
    }

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_BITSET_H__INCLUDED
