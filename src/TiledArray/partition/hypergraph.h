/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  hypergraph.h
 *  Sep 11, 2015
 *
 */

#ifndef TILEDARRAY_PARTITION_HYPERGRAPH_H__INCLUDED
#define TILEDARRAY_PARTITION_HYPERGRAPH_H__INCLUDED

// Undefine MIN and MAX to avoid conflicts between definitions in Mondriaan and
// other system headers, specifically on OS X.
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
extern "C" {
#include <Mondriaan.h>
}
#include <TiledArray/error.h>

namespace TiledArray {
  namespace detail {

    class HyperGraph {
    public:
      typedef long size_type;
    private:
      std::shared_ptr<sparsematrix> hypergraph_;

    public:
      /// Compiler generated constructors constructor
      HyperGraph() = default;
      HyperGraph(const HyperGraph&) = default;
      HyperGraph(HyperGraph&&) = default;
      HyperGraph& operator=(const HyperGraph&) = default;
      HyperGraph& operator=(HyperGraph&&) = default;

      ///
      HyperGraph(const size_type num_cells, const size_type num_nets,
          const size_type max_pins) :
          hypergraph_(std::make_shared<sparsematrix>())
      {
        // The code in this constructor is based CRSSparseMatrixNetWeightInit in
        // Mondriaan 4.

        // Initialize hypergraph_ with zero.
        if(! MMSparseMatrixInit(hypergraph_.get()))
          TA_EXCEPTION("Mondriaan: Could not initialize sparse matrix");

        // The possibilities for the MMTypeCode are:
        //
        //   Object = matrix, distributed matrix, or weighted matrix
        //   TypeCode[0] = M, D, W
        //
        //   Format = coordinate, array
        //   TypeCode[1] = C, A
        //
        //   Field =  integer, real, complex, or pattern
        //   TypeCode[2] = I, R, C, P
        //
        //   Symmetry = general, symmetric, skew-symmetric, or hermitian
        //   TypeCode[3] = G, S, K, H
        hypergraph_->MMTypeCode[ 0 ] = 'M';
        hypergraph_->MMTypeCode[ 1 ] = 'C';
        hypergraph_->MMTypeCode[ 2 ] = 'R';
        hypergraph_->MMTypeCode[ 3 ] = 'G';

        // Initialize the hypergraph dimensions
        hypergraph_->m = num_nets;
        hypergraph_->n = num_cells;
        hypergraph_->NrNzElts = max_pins;

        // Allocate memory for hypergraph based on the above initializations
        if(! MMSparseMatrixAllocateMemory(hypergraph_.get()))
          TA_EXCEPTION("Mondriaan: Could not allocate sparse matrix" );

        // Allocate memory for and initialize weight data
        if(! MMWeightsInit(hypergraph_.get()))
          TA_EXCEPTION("Error during row/column weight initialization");
        memset(hypergraph_->RowWeights, 0, num_nets * sizeof(hypergraph_->RowWeights));
        memset(hypergraph_->ColWeights, 0, num_cells * sizeof(hypergraph_->ColWeights));

        // Set the number of pins to 0. Calling add_pin will increment the value.
        hypergraph_->NrNzElts = 0l;
      }

      ~HyperGraph() {
        if(hypergraph_)
          MMDeleteSparseMatrix(hypergraph_.get());
      }

      /// Add pin to hyper graph

      /// \param pin The pin id
      /// \param cell The cell of the pin
      /// \param net The net of the pin
      /// \param weight The weight of the pin
      void add_pin(const size_type cell, const size_type net, long weight) {
        TA_ASSERT(hypergraph_);
        TA_ASSERT(cell < hypergraph_->n);
        TA_ASSERT(net < hypergraph_->m);

        const long pin = hypergraph_->NrNzElts++;

        // Set pin data
        hypergraph_->i[pin] = net;
        hypergraph_->j[pin] = cell;
        hypergraph_->ReValue[pin] = weight;

        // Add pin weight to column
        hypergraph_->ColWeights[cell] += weight;
      }

      /// Pin count accessor

      /// \return Number of pins
      size_type pins() const {
        TA_ASSERT(hypergraph_);
        return hypergraph_->NrNzElts;
      }

      /// Cells accessor

      /// \return The number of cells in the graph
      size_type cells() const {
        TA_ASSERT(hypergraph_);
        return hypergraph_->n;
      }

      /// Nets accessor

      /// \return The number of nets in the graph
      size_type nets() const {
        TA_ASSERT(hypergraph_);
        return hypergraph_->m;
      }

      /// Const cell weight accessor

      /// \return A const pointer to the net weights array
      const long* cell_weights() const {
        TA_ASSERT(hypergraph_);
        return hypergraph_->ColWeights;
      }

      /// Cell weight accessor

      /// \return A pointer to the net weights array
      long* cell_weights() {
        TA_ASSERT(hypergraph_);
        return hypergraph_->ColWeights;
      }

      /// Const net weight accessor

      /// \return A const pointer to the net weights array
      const long* net_weights() const {
        TA_ASSERT(hypergraph_);
        return hypergraph_->RowWeights;
      }

      /// Net weight accessor

      /// \return A pointer to the net weights array
      long* net_weights() {
        TA_ASSERT(hypergraph_);
        return hypergraph_->RowWeights;
      }

      /// Partition the hypergraph

      /// \param k The number of partitions
      /// \param seed The seed value to the random number generator (default = 0)
      /// \param epsilon The maximum allowed load imbalance (default = 0.03 or 3%)
      void partition(const size_type k, const long seed = 0l, const double epsilon = 0.03) {
        TA_ASSERT(hypergraph_);

        // Initialize the Mondriaan options object that controls partitioning.
        opts options;
        SetDefaultOptions(&options);
        SetOption(&options, "SplitStrategy", "onedimcol");
        if(! ApplyOptions(&options))
          TA_EXCEPTION("Invalid options");

        // Set the random number generator seed value (over rides the value set
        // in ApplyOptions).
        options.Seed = seed;

        // Initialize the partition information arrays
        if(! PstartInit(hypergraph_.get(), k)) {
          TA_EXCEPTION("error during initialisation of partition!\n");
        }

        // Distribute the matrix over k processors with an allowed imbalance
        // of epsilon and the options provided above.
        if(! DistributeMatrixMondriaan(hypergraph_.get(), k, epsilon, &options, NULL))
          TA_EXCEPTION("Unable to distribute hypergraph");


        // Output: The nonzeros of partition i are in positions
        //         [ hypergraph_->Pstart[i], hypergraph_->Pstart[i+1] ).
        //         The values in Pstart are a map to the hypergraph_ vectors
        //         hypergraph_->i, hypergraph_->j, and hypergraph_->ReValue.
        //         The row partitions are in the j vector.
      }

      std::vector<size_type> get_partition_map() const {
        TA_ASSERT(hypergraph_);
        TA_ASSERT(hypergraph_->NrProcs > 0l);
        TA_ASSERT(hypergraph_->Pstart != nullptr);

        // Get the number of partitions and cells
        const size_type num_parts = hypergraph_->NrProcs;

        std::vector<size_type> result;
        result.reserve(hypergraph_->n);

        for(size_type part = 0l; part < num_parts; ++part) {
          // Get the start and end of the part range
          size_type part_first = hypergraph_->Pstart[part];
          const size_type part_last = hypergraph_->Pstart[part + 1l];

          const size_type part_start = result.size();

          for(; part_first < part_last; ++part_first) {
            const size_type cell = hypergraph_->j[part_first];

            // Add unique cells to the partition map
            if(std::find(result.begin() + part_start, result.end(), cell) == result.end())
              result.emplace_back(cell);
          }
        }

        return result;
      }

      /// Compute the total communication volume
      long cut_set() const {
        TA_ASSERT(hypergraph_);
        TA_ASSERT(hypergraph_->NrProcs > 0l);

        // Get the number of partitions per net
        const long num_nets = hypergraph_->m;
        std::unique_ptr<int, decltype(&free)>
        parts_per_net(static_cast<int*>(malloc(num_nets * sizeof(int))), &free);
        if(! InitNprocs(hypergraph_->get(), COL, parts_per_net.get())) {
          TA_EXCEPTION("Unable to initialize processor array");
        }

        // Compute the total communication volume, which is equal to
        // comm_volume = \sum_j w_j * (np_j - 1)
        long comm_volume = 0l;
        math::reduce_op([] (long& c, const long w, const int np)
            { c += w * std::max(np - 1, 0); }, num_nets, comm_volume,
            hypergraph_->RowWeights, parts_per_net.get());

        return comm_volume;
      }

    }; // class HyperGraph

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_PARTITION_HYPERGRAPH_H__INCLUDED
