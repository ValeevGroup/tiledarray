//============================================================================
// Name        : CCD.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   :
// Description : Hello World in C, Ansi-style
//============================================================================

#include <tiled_array.h>
#include <fstream>
#include <sstream>

using namespace TiledArray;


class InputData {
public:
  typedef std::vector<std::size_t> obs_mosym;
  typedef std::vector<std::pair<std::array<std::size_t, 2>, double> > array2d;
  typedef std::vector<std::pair<std::array<std::size_t, 4>, double> > array4d;

private:
  std::string name_;
  unsigned long nirreps_;
  unsigned long nmo_;
  unsigned long nocc_act_alpha_;
  unsigned long nocc_act_beta_;
  unsigned long nvir_act_alpha_;
  unsigned long nvir_act_beta_;
  obs_mosym obs_mosym_alpha_;
  obs_mosym obs_mosym_beta_;
  array2d f_;
  array4d v_ab_;

  template <typename I>
  struct predicate {
    typedef bool result_type;

    predicate(const I& i) : index_(i) { }

    template <typename V>
    result_type operator()(const std::pair<I,V>& data) const { return data.first == index_; }

  private:
    I index_;
  };
public:

  InputData(std::ifstream& input) {
    std::string lable;
    input >> lable >> name_;
//    std::cout << lable << name_ << "\n";
    input >> lable >> nirreps_;
//    std::cout << lable << " " << nirreps_ << "\n";
    input >> lable >> nmo_;
//    std::cout << lable << " " << nmo_ << "\n";
    input >> lable >> nocc_act_alpha_;
//    std::cout << lable << " " << nocc_act_alpha_ << "\n";
    input >> lable >> nocc_act_beta_;
//    std::cout << lable << " " << nocc_act_beta_ << "\n";
    input >> lable >> nvir_act_alpha_;
//    std::cout << lable << " " << nvir_act_alpha_ << "\n";
    input >> lable >> nvir_act_beta_;
//    std::cout << lable << " " << nvir_act_beta_ << "\n";
    input >> lable;
//    std::cout << lable << "\n";
    obs_mosym_alpha_.resize(nmo_, 0);
    for(obs_mosym::iterator it = obs_mosym_alpha_.begin(); it != obs_mosym_alpha_.end(); ++it) {
      input >> *it;
//      std::cout << *it << "\n";
    }
    input >> lable;
//    std::cout << lable << "\n";
    obs_mosym_beta_.resize(nmo_, 0);
    for(obs_mosym::iterator it = obs_mosym_beta_.begin(); it != obs_mosym_beta_.end(); ++it) {
      input >> *it;
//      std::cout << *it << "\n";
    }
    std::string line;
    std::getline(input, line);
    std::getline(input, line);
    do {
      line.clear();
      std::getline(input, line);
      if(line.size() == 0ul)
        break;
      std::istringstream iss(line);
      array2d::value_type data;
      iss >> data.first[0] >> data.first[1] >> data.second;
      f_.push_back(data);
//      std::cout << "(" << data.first[0] << ", " << data.first[1] << ") " << data.second << "\n";
    } while(! input.eof());
    do {
      line.clear();
      std::getline(input, line);
      if(line.size() == 0ul)
        break;
      std::istringstream iss(line);
      array4d::value_type data;
      iss >> data.first[0] >> data.first[1] >> data.first[2] >> data.first[3] >> data.second;
      v_ab_.push_back(data);
//      std::cout << "(" << data.first[0] << ", " << data.first[1] << ", " << data.first[2]
//          << ", " << data.first[3] << ") " << data.second << "\n";
    } while(! input.eof());
  }

  std::string name() const { return name_; }
  unsigned long nirreps() const { return nirreps_; }
  unsigned long nmo() const { return nmo_; }
  unsigned long nocc_act_alpha() const { return nocc_act_alpha_; }
  unsigned long nocc_act_beta() const { return nocc_act_beta_; }
  unsigned long nvir_act_alpha() const { return nvir_act_alpha_; }
  unsigned long nvir_act_beta() const { return nvir_act_beta_; }
  std::size_t obs_mosym_alpha(std::size_t i) const { return obs_mosym_alpha_[i]; }
  std::size_t obs_mosym_beta(std::size_t i) const { return obs_mosym_beta_[i]; }

  double f(std::size_t i, std::size_t j) const {
    std::array<std::size_t, 2> index = {{ i, j }};
    array2d::const_iterator it = std::find_if(f_.begin(), f_.end(),
        predicate<std::array<std::size_t, 2> >(index));
    if(it != f_.end())
      return it->second;

    return 0.0;
  }

  const array2d& f() const { return f_; }

  double v_ab(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
    std::array<std::size_t, 4> index = {{ i, j, k, l }};
    array4d::const_iterator it = std::find_if(v_ab_.begin(), v_ab_.end(),
        predicate<std::array<std::size_t, 4> >(index));
    if(it != v_ab_.end())
      return it->second;

    return 0.0;
  }

  const array4d& v_ab() const { return v_ab_; }
};


void solve_t(madness::World& world, Array<double, CoordinateSystem<4> >& t,
    const Array<double, CoordinateSystem<2> >& f, const Array<double, CoordinateSystem<4> >& v) {
  double r0 = 0;
  double r = 1;



    world.gop.fence();


}

template <typename Range>
std::vector<std::size_t> make_sparse_list(const Range& range) {
  std::vector<std::size_t> result;
  result.reserve(range.volume());

}

int main(int argc, char** argv) {
  // Initialize madness runtime
  madness::initialize(argc,argv);
  {
    madness::World world(MPI::COMM_WORLD);

    // Get input data
    std::ifstream input("input");
    InputData data(input);
    input.close();

    // Construct tiled range objects
    std::array<TiledRange1, 4> tr_list = {{
        TiledRange1(0, 4, 0, data.nocc_act_alpha(), data.nmo(), data.nmo() + data.nocc_act_beta(), 2 * data.nmo()),
        TiledRange1(0, 4, 0, data.nocc_act_alpha(), data.nmo(), data.nmo() + data.nocc_act_beta(), 2 * data.nmo()),
        TiledRange1(0, 4, 0, data.nocc_act_alpha(), data.nmo(), data.nmo() + data.nocc_act_beta(), 2 * data.nmo()),
        TiledRange1(0, 4, 0, data.nocc_act_alpha(), data.nmo(), data.nmo() + data.nocc_act_beta(), 2 * data.nmo()) }};
    StaticTiledRange<CoordinateSystem<2> > tr2(tr_list.begin(), tr_list.begin() + 2);
    StaticTiledRange<CoordinateSystem<4> > tr4(tr_list.begin(), tr_list.end());

    // Construct f array
    std::array<std::size_t, 5> f_sparse_list = {{ 0, 5, 10, 15 }};
    Array<double, CoordinateSystem<2> > f(world, tr2, f_sparse_list.begin(), f_sparse_list.end());

    // Initialize f tiles
    for(std::array<std::size_t, 2>::const_iterator it = f_sparse_list.begin(); it != f_sparse_list.begin() + (f_sparse_list.size() / 2); ++it) {
      if(f.is_local(*it))
        f.set(*it, 0.0);
    }

    if(f.is_local(f_sparse_list[0]))
      f.set(f_sparse_list[2], f.find(f_sparse_list[0]));
    if(f.is_local(f_sparse_list[0]))
      f.set(f_sparse_list[2], f.find(f_sparse_list[0]));

    for(InputData::array2d::const_iterator it = data.f().begin(); it != data.f().end(); ++it) {
      Array<double, CoordinateSystem<2> >::range_type::index index = f.trange().element_to_tile(it->first);
      if(f.is_local(index))
        f.find(index).get()[it->first];
      index[0] += data.nmo();
    }


    Array<double, CoordinateSystem<4> > v_ab(world, tr4);
    for(Array<double, CoordinateSystem<4> >::range_type::const_iterator it = v_ab.range().begin(); it != v_ab.range().end(); ++it) {
      if(v_ab.is_local(*it))
        v_ab.set(*it, 0.0);
    }

    for(InputData::array4d::const_iterator it = data.v_ab().begin(); it != data.v_ab().end(); ++it) {
      Array<double, CoordinateSystem<4> >::range_type::index index = v_ab.trange().element_to_tile(it->first);
      if(v_ab.is_local(index))
        v_ab.find(index).get()[it->first];
    }

    // Fence to make sure data on all nodes has been initialized
    world.gop.fence();

    Array<double, CoordinateSystem<4> > v_aa = v_ab("a,b,c,d") - v_ab("a,b,d,c");
    Array<double, CoordinateSystem<4> >& v_bb = v_aa;

  }

  std::cout << "Done!\n";
  // stop the madenss runtime
  madness::finalize();
	return 0;
}
