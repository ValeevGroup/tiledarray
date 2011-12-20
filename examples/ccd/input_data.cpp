#include "input_data.h"
#include <fstream>
#include <sstream>

TiledArray::TiledRange1 InputData::make_trange1(obs_mosym::const_iterator first, obs_mosym::const_iterator last) {
  std::vector<std::size_t> tiles;
  tiles.push_back(0ul);
  obs_mosym::value_type current = *first;

  for(obs_mosym::const_iterator it = first; it != last; ++it) {
    if(*it != current) {
      tiles.push_back(std::distance(first, it));
      current = *it;
    }
  }

  return TiledArray::TiledRange1(tiles.begin(), tiles.end());
}

TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> >
InputData::trange(const Spin s, const RangeOV ov1, const RangeOV ov2) const {

  const obs_mosym& spin = (s == alpha ? obs_mosym_alpha_ : obs_mosym_beta_);
  const std::size_t& nocc = (s == alpha ? nocc_act_alpha_ : nocc_act_beta_);
  const std::size_t first1 = (ov1 == occ ? 0 : nocc);
  const std::size_t last1 = (ov1 == occ ? nocc : nmo_);
  const std::size_t first2 = (ov2 == occ ? 0 : nocc);
  const std::size_t last2 = (ov2 == occ ? nocc : nmo_);

  const std::array<TiledArray::TiledRange1, 2> tr_list = {{
      make_trange1(spin.begin() + first1, spin.begin() + last1),
      make_trange1(spin.begin() + first2, spin.begin() + last2) }};

  return TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> >(tr_list.begin(), tr_list.end());
}

TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<4> >
InputData::trange(const Spin s1, const Spin s2, const RangeOV ov1, const RangeOV ov2, const RangeOV ov3, const RangeOV ov4) const {

  const obs_mosym& spin1 = (s1 == alpha ? obs_mosym_alpha_ : obs_mosym_beta_);
  const std::size_t& nocc1 = (s1 == alpha ? nocc_act_alpha_ : nocc_act_beta_);
  const obs_mosym& spin2 = (s2 == alpha ? obs_mosym_alpha_ : obs_mosym_beta_);
  const std::size_t& nocc2 = (s2 == alpha ? nocc_act_alpha_ : nocc_act_beta_);
  const std::size_t first1 = (ov1 == occ ? 0 : nocc1);
  const std::size_t last1 = (ov1 == occ ? nocc1 : nmo_);
  const std::size_t first2 = (ov2 == occ ? 0 : nocc2);
  const std::size_t last2 = (ov2 == occ ? nocc2 : nmo_);
  const std::size_t first3 = (ov3 == occ ? 0 : nocc1);
  const std::size_t last3 = (ov3 == occ ? nocc1 : nmo_);
  const std::size_t first4 = (ov4 == occ ? 0 : nocc2);
  const std::size_t last4 = (ov4 == occ ? nocc2 : nmo_);

  const std::array<TiledArray::TiledRange1, 4> tr_list = {{
      make_trange1(spin1.begin() + first1, spin1.begin() + last1),
      make_trange1(spin2.begin() + first2, spin2.begin() + last2),
      make_trange1(spin1.begin() + first3, spin1.begin() + last3),
      make_trange1(spin2.begin() + first4, spin2.begin() + last4) }};

  return TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<4> >(tr_list.begin(), tr_list.end());
}

InputData::InputData(std::ifstream& input) {
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

double InputData::f(std::size_t i, std::size_t j) const {
  std::array<std::size_t, 2> index = {{ i, j }};
  array2d::const_iterator it = std::find_if(f_.begin(), f_.end(),
      predicate<std::array<std::size_t, 2> >(index));
  if(it != f_.end())
    return it->second;

  return 0.0;
}

double InputData::v_ab(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
  std::array<std::size_t, 4> index = {{ i, j, k, l }};
  array4d::const_iterator it = std::find_if(v_ab_.begin(), v_ab_.end(),
      predicate<std::array<std::size_t, 4> >(index));
  if(it != v_ab_.end())
    return it->second;

  return 0.0;
}

TiledArray::Array<double, TiledArray::CoordinateSystem<2> >
InputData::make_f(madness::World& w, const Spin s, const RangeOV ov1, const RangeOV ov2) {
  // Construct the array
  TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> > tr = trange(s, ov1, ov2);
  std::vector<std::size_t> sparse_list = make_sparse_list(tr, f_);
  TiledArray::Array<double, TiledArray::CoordinateSystem<2> > f(w, tr, sparse_list.begin(), sparse_list.end());

  // Initialize tiles
  for(std::vector<std::size_t>::const_iterator it = sparse_list.begin(); it != sparse_list.end(); ++it) {
    if(f.is_local(*it))
      f.set(*it, 0.0);
  }

  // Set the tile data
  TiledArray::Array<double, TiledArray::CoordinateSystem<2> >::range_type::index index;
  for(array2d::const_iterator it = f_.begin(); it != f_.end(); ++it) {
    index = f.trange().element_to_tile(it->first);
    if(f.is_local(index))
      f.find(index).get()[it->first] = it->second;
  }

  return f;
}

TiledArray::Array<double, TiledArray::CoordinateSystem<4> >
InputData::make_v_ab(madness::World& w, const RangeOV ov1, const RangeOV ov2, const RangeOV ov3, const RangeOV ov4) {
  // Construct the array
  TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<4> > tr = trange(alpha, beta, ov1, ov2, ov3, ov4);
  std::vector<std::size_t> sparse_list = make_sparse_list(tr, v_ab_);
  TiledArray::Array<double, TiledArray::CoordinateSystem<4> > v_ab(w, tr, sparse_list.begin(), sparse_list.end());

  // Initialize tiles
  for(std::vector<std::size_t>::const_iterator it = sparse_list.begin(); it != sparse_list.end(); ++it) {
    if(v_ab.is_local(*it))
      v_ab.set(*it, 0.0);
  }

  // Set the tile data
  TiledArray::Array<double, TiledArray::CoordinateSystem<4> >::range_type::index index;
  for(array4d::const_iterator it = v_ab_.begin(); it != v_ab_.end(); ++it) {
    index = v_ab.trange().element_to_tile(it->first);
    if(v_ab.is_local(index))
      v_ab.find(index).get()[it->first] = it->second;
  }

  return v_ab;
}
