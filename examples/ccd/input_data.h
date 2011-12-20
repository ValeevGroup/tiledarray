#ifndef EXAMPLES_CCD_INPUT_DATA_H__INCLUDED
#define EXAMPLES_CCD_INPUT_DATA_H__INCLUDED

#include <vector>
#include <iosfwd>
#include <world/stdarray.h>
#include <tiled_array.h>

typedef enum {
  alpha,
  beta
} Spin;

typedef enum {
  occ,
  vir
} RangeOV;

class InputData {
public:
  typedef std::vector<std::size_t> obs_mosym;
  typedef std::vector<std::pair<std::array<std::size_t, 2>, double> > array2d;
  typedef std::vector<std::pair<std::array<std::size_t, 4>, double> > array4d;
  typedef TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<2> > tr2;
  typedef TiledArray::StaticTiledRange<TiledArray::CoordinateSystem<4> > tr4;

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

  static TiledArray::TiledRange1 make_trange1(obs_mosym::const_iterator first,
      obs_mosym::const_iterator last);

public:

  InputData(std::ifstream& input);

  std::string name() const { return name_; }
//  unsigned long nirreps() const { return nirreps_; }
//  unsigned long nmo() const { return nmo_; }
//  unsigned long nocc_act_alpha() const { return nocc_act_alpha_; }
//  unsigned long nocc_act_beta() const { return nocc_act_beta_; }
//  unsigned long nvir_act_alpha() const { return nvir_act_alpha_; }
//  unsigned long nvir_act_beta() const { return nvir_act_beta_; }
//  obs_mosym obs_mosym_alpha() const { return obs_mosym_alpha_; }
//  obs_mosym obs_mosym_beta() const { return obs_mosym_beta_; }


  double f(std::size_t i, std::size_t j) const;

  const array2d& f() const { return f_; }

  double v_ab(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const;

  const array4d& v_ab() const { return v_ab_; }

  tr2 trange(const Spin s, const RangeOV ov1, const RangeOV ov2) const;

  tr4 trange(const Spin s1, const Spin s2, const RangeOV ov1, const RangeOV ov2,
      const RangeOV ov3, const RangeOV ov4) const;

  template <typename R, typename T>
  std::vector<std::size_t> make_sparse_list(const R& r, const T& t) const {
    std::vector<std::size_t> result;

    // Find and store the tile for each element in the tensor.
    for(typename T::const_iterator it = t.begin(); it != t.end(); ++it)
      if(r.elements().includes(it->first))
        result.push_back(r.tiles().ord(r.element_to_tile(it->first)));


    // Remove duplicates.
    std::sort(result.begin(), result.end());
    result.resize(std::distance(result.begin(), std::unique(result.begin(), result.end())));

    return result;
  }

  TiledArray::Array<double, TiledArray::CoordinateSystem<2> >
  make_f(madness::World& w, const Spin s, const RangeOV ov1, const RangeOV ov2);

  TiledArray::Array<double, TiledArray::CoordinateSystem<4> >
  make_v_ab(madness::World& w, const RangeOV ov1, const RangeOV ov2, const RangeOV ov3, const RangeOV ov4);
};


#endif // EXAMPLES_CCD_INPUT_DATA_H__INCLUDED
