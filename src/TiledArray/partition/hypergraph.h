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

    class HyperGraph;

    class PartInfo {
    public:
      std::vector<long> part_start;///< procstart_[j] is the position in array
                              ///< procindex_ of the first processor number of
                              ///< net j, 0 <= j < hypergraph->m.
                              ///< procstart[hypergraph->m] = total number of
                              ///< processor numbers of all the columns
                              ///< together.
      std::vector<int> cellindex; ///< contains the processor numbers
                              ///< occurring in the matrix columns. The
                              ///< processor numbers for one column j are
                              ///< stored consecutively.

      // Compiler generated constructors/destructor
      PartInfo() = default;
      PartInfo(const PartInfo&) = default;
      PartInfo(PartInfo&&) = default;
      PartInfo& operator=(const PartInfo&) = default;
      PartInfo& operator=(PartInfo&&) = default;
      ~PartInfo() = default;


      friend std::ostream& operator<<(std::ostream& os, const PartInfo& info) {
        for(long p = 1l; p < info.part_start.size(); ++p) {
          long first = info.part_start[p - 1];
          const long last = info.part_start[p];
          os  << "{";
          for(; first < last; ++first)
            os << " " << info.cellindex[first];
          os << " } ";
        }
        return os;
      }

    };

    class MatrixPartInfo {

      PartInfo row_part_; ///< Row partitioning information
      PartInfo col_part_; ///< Column partitioning information

    public:
      // Compiler generated constructors/destructor
      MatrixPartInfo() = default;
      MatrixPartInfo(const MatrixPartInfo&) = default;
      MatrixPartInfo(MatrixPartInfo&&) = default;
      MatrixPartInfo& operator=(const MatrixPartInfo&) = default;
      MatrixPartInfo& operator=(MatrixPartInfo&&) = default;
      ~MatrixPartInfo() = default;

      inline void set_row_partition(const HyperGraph& hg);

      inline void set_col_partition(const HyperGraph& hg);

      friend std::ostream& operator<<(std::ostream& os, const MatrixPartInfo& info) {
        os << info.row_part_ << "\n" << info.col_part_ << "\n";
        return os;
      }
    };

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
        hypergraph_->MMTypeCode[ 0 ] = 'W';
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
//        SetOption(&options, "Coarsening_NrVertices", "4");

#ifdef TILEDARRAY_HAS_PATOH
        SetOption(&options, "Partitioner", "patoh");
//        SetOption(&options, "Partitioner", "fullpatoh");
#endif
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

      PartInfo get_partition_map() const {
        TA_ASSERT(hypergraph_);
        TA_ASSERT(hypergraph_->NrProcs > 0l);
        TA_ASSERT(hypergraph_->Pstart != nullptr);

        const int num_parts = hypergraph_->NrProcs;
        const long num_cells = hypergraph_->n;

        // Allocate memory for the result partition info object
        PartInfo result;
        result.part_start.reserve(num_parts + 1);
        result.cellindex.reserve(num_cells);

        for(size_type part = 0l; part < num_parts; ++part) {
          // Get the start and end of the part range from the hypergraph
          size_type part_first = hypergraph_->Pstart[part];
          const size_type part_last = hypergraph_->Pstart[part + 1l];

          const size_type part_start = result.cellindex.size();
          result.part_start.emplace_back(part_start);

          // Collect add all cells that occur within part to the list
          for(; part_first < part_last; ++part_first) {
            const size_type cell = hypergraph_->j[part_first];

            // Add unique cells to the partition map
            if(std::find(result.cellindex.begin() + part_start,
                result.cellindex.end(), cell) == result.cellindex.end())
              result.cellindex.emplace_back(cell);
          }
        }

        return result;
      }

      /// Compute the cut set
      long cut_set() const {
        TA_ASSERT(hypergraph_);
        TA_ASSERT(hypergraph_->NrProcs > 0l);

        // Compute the total communication volume, which is equal to
        // comm_volume = \sum_j w_j * (np_j - 1)
        long cut_size = 0l;
        math::reduce_op([] (long& c, const long weight, const int lambda)
            { c += weight * std::max(lambda - 1, 0); }, hypergraph_->m, cut_size,
            hypergraph_->RowWeights, hypergraph_->RowLambda);

//        for(long net = 0ul; net < hypergraph_->m; ++net) {
//          const long weight = hypergraph_->RowWeights[net];
//          long lambda = hypergraph_->RowLambda[net];
//
//          std::cout << "  " << net << "  " << lambda << "  " << weight << "\n";
//        }

        return cut_size;
      }


      /// Compute the initial cut set

      /// Here we assume a round robin distribution for the initial
      /// partitioning.
      long init_cut_set(long num_parts) const {
        TA_ASSERT(hypergraph_);

        const long num_nets = hypergraph_->m;
        const long num_pins = hypergraph_->NrNzElts;
        std::vector<long> parts_per_net(num_nets * num_parts, 0);

        for(long pin = 0l; pin < num_pins; ++pin) {
          const long cell = hypergraph_->j[pin];
          const long net = hypergraph_->i[pin];
          const long part = cell % num_parts;
          parts_per_net[net * num_parts + part] |= 1l;
        }

        long cut_size = 0l;
        for(long net = 0; net < num_nets; ++net) {
          const long weight = hypergraph_->RowWeights[net];
          long lambda = 0l;
          for(long part = 0l; part < num_parts; ++part)
            lambda += parts_per_net[net * num_parts + part];

//          std::cout << "  " << net << "  " << lambda << "  " << weight << "\n";
          cut_size += weight * std::max(lambda - 1l, 0l);
        }

        return cut_size;
      }

      void verify_lambdas() const {
        TA_ASSERT(hypergraph_);

        const auto* lambdas = hypergraph_->RowLambda;
        const auto num_nets = hypergraph_->m;
        const auto num_parts  = hypergraph_->NrProcs;
        const auto num_pins = hypergraph_->NrNzElts;

        /* Verification function of the calculated lambdas. */
        long max_lambda = num_parts + 1;
        std::vector<long> hist(max_lambda, 0l);
        std::vector<long> ihist(max_lambda, 0l);
        std::vector<long> parts_per_net(num_nets * num_parts, 0);


        for(long t = 0; t < num_nets; t++) {
          const long l = lambdas[t];
          ++hist[l];
        }

        for(long pin = 0l; pin < num_pins; ++pin) {
          const long cell = hypergraph_->j[pin];
          const long net = hypergraph_->i[pin];
          const long part = cell % num_parts;
          parts_per_net[net * num_parts + part] |= 1l;
        }

        long cut_size = 0l;
        for(long net = 0; net < num_nets; ++net) {
          const long weight = hypergraph_->RowWeights[net];
          long lambda = 0l;
          for(long part = 0l; part < num_parts; ++part)
            lambda += parts_per_net[net * num_parts + part];
          ++ihist[lambda];
        }


        std::cout << "lambda histogram:\n";
        for(long t = 0; t < max_lambda; t++)
          std::cout << "    " << t << "\t" << ihist[t] << "\t" << hist[t] << "\n";

      }

    }; // class HyperGraph


    inline void MatrixPartInfo::set_row_partition(const HyperGraph& hg) {
      row_part_ = hg.get_partition_map();
    }

    inline void MatrixPartInfo::set_col_partition(const HyperGraph& hg) {
      col_part_ = hg.get_partition_map();
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_PARTITION_HYPERGRAPH_H__INCLUDED
