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

#include "input_data.h"
#include <fstream>
#include <sstream>

TiledArray::TiledRange1
InputData::make_trange1(const obs_mosym::const_iterator& begin, obs_mosym::const_iterator first, obs_mosym::const_iterator last) {
  std::vector<std::size_t> tiles;
  obs_mosym::value_type current = *first;
  tiles.push_back(std::distance(begin, first));

  for(; first != last; ++first) {
    if(*first != current) {
      tiles.push_back(std::distance(begin, first));
      current = *first;
    }
  }

  tiles.push_back(std::distance(begin, first));

  return TiledArray::TiledRange1(tiles.begin(), tiles.end());
}

TArray2s::trange_type
InputData::trange(const Spin s, const RangeOV ov1, const RangeOV ov2) const {

  const obs_mosym& spin = (s == alpha ? obs_mosym_alpha_ : obs_mosym_beta_);
  const std::size_t nocc = (s == alpha ? nocc_act_alpha_ : nocc_act_beta_);
  const std::size_t first1 = (ov1 == occ ? 0 : nocc);
  const std::size_t last1 = (ov1 == occ ? nocc : nmo_);
  const std::size_t first2 = (ov2 == occ ? 0 : nocc);
  const std::size_t last2 = (ov2 == occ ? nocc : nmo_);

  const std::array<TiledArray::TiledRange1, 2> tr_list = {{
      make_trange1(spin.begin(), spin.begin() + first1, spin.begin() + last1),
      make_trange1(spin.begin(), spin.begin() + first2, spin.begin() + last2) }};

  return TiledArray::TiledRange(tr_list.begin(), tr_list.end());
}

TArray4s::trange_type
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
      make_trange1(spin1.begin(), spin1.begin() + first1, spin1.begin() + last1),
      make_trange1(spin2.begin(), spin2.begin() + first2, spin2.begin() + last2),
      make_trange1(spin1.begin(), spin1.begin() + first3, spin1.begin() + last3),
      make_trange1(spin2.begin(), spin2.begin() + first4, spin2.begin() + last4) }};

  return TiledArray::TiledRange(tr_list.begin(), tr_list.end());
}

InputData::InputData(std::ifstream& input) {
  std::string label;
  input >> label >> name_;
//  std::cout << label << name_ << "\n";
  input >> label >> nirreps_;
//  std::cout << label << " " << nirreps_ << "\n";
  input >> label >> nmo_;
//  std::cout << label << " " << nmo_ << "\n";
  input >> label >> nocc_act_alpha_;
//  std::cout << label << " " << nocc_act_alpha_ << "\n";
  input >> label >> nocc_act_beta_;
//  std::cout << label << " " << nocc_act_beta_ << "\n";
  input >> label >> nvir_act_alpha_;
//  std::cout << label << " " << nvir_act_alpha_ << "\n";
  input >> label >> nvir_act_beta_;
//  std::cout << label << " " << nvir_act_beta_ << "\n";
  input >> label;
//  std::cout << label << "\n";
  obs_mosym_alpha_.resize(nmo_, 0);
  for(obs_mosym::iterator it = obs_mosym_alpha_.begin(); it != obs_mosym_alpha_.end(); ++it) {
    input >> *it;
//    std::cout << *it << "\n";
  }
  input >> label;
//  std::cout << label << "\n";
  obs_mosym_beta_.resize(nmo_, 0);
  for(obs_mosym::iterator it = obs_mosym_beta_.begin(); it != obs_mosym_beta_.end(); ++it) {
    input >> *it;
//    std::cout << *it << "\n";
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

    // Note: Input data is in chemist notation order, but we want physicist notation.
    // So we swap index 1 and 2
    iss >> data.first[0] >> data.first[2] >> data.first[1] >> data.first[3] >> data.second;
    v_ab_.push_back(data);
//      std::cout << "(" << data.first[0] << ", " << data.first[1] << ", " << data.first[2]
//          << ", " << data.first[3] << ") " << data.second << "\n";
  } while(! input.eof());
}

TArray2s
InputData::make_f(madness::World& w, const Spin s, const RangeOV ov1, const RangeOV ov2) {
  // Construct the array
  TiledArray::TiledRange tr = trange(s, ov1, ov2);
//  std::cout << tr << "\n";

  TArray2s f(w, tr, make_sparse_shape(tr, f_));

  // Initialize tiles
  f.set_all_local(0.0);

  // Set the tile data
  TArray2s::range_type::index index;
  for(array2d::const_iterator it = f_.begin(); it != f_.end(); ++it) {
    if(f.trange().elements().includes(it->first)) {
      index = f.trange().element_to_tile(it->first);
      if(f.is_local(index))
        f.find(index).get()[it->first] = it->second;
    }
  }

  return f;
}

TArray4s
InputData::make_v_ab(madness::World& w, const RangeOV ov1, const RangeOV ov2, const RangeOV ov3, const RangeOV ov4) {
  // Construct the array
  TiledArray::TiledRange tr = trange(alpha, beta, ov1, ov2, ov3, ov4);
//  std::cout << tr << "\n";
  TArray4s v_ab(w, tr,make_sparse_shape(tr, v_ab_));

  // Initialize tiles
  v_ab.set_all_local(0.0);

  // Set the tile data
  TArray4s::range_type::index index;
  for(array4d::const_iterator it = v_ab_.begin(); it != v_ab_.end(); ++it) {
    if(v_ab.trange().elements().includes(it->first)) {
      index = v_ab.trange().element_to_tile(it->first);
      if(v_ab.is_local(index))
        v_ab.find(index).get()[it->first] = it->second;
    }
  }

  return v_ab;
}
