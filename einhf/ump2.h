/*
 * @BEGIN LICENSE
 *
 * einhf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Psi4; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#pragma once

#include "einsums.hpp"
#include <deque>
#include <vector>

#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"
#include "rhf.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

struct UMP2ScaleFunction
    : public virtual einsums::tensor_props::FunctionTensorBase<double, 4>,
      virtual einsums::tensor_props::CoreTensorBase {
private:
  const einsums::Tensor<double, 1> *_aevals, *_bevals;

public:
  UMP2ScaleFunction() = default;
  UMP2ScaleFunction(const UMP2ScaleFunction &) = default;

  UMP2ScaleFunction(std::string name, const einsums::Tensor<double, 1> *aevals, const einsums::Tensor<double, 1> *bevals)
      : einsums::tensor_props::FunctionTensorBase<double, 4>(
            name, aevals->dim(0), aevals->dim(0), aevals->dim(0), aevals->dim(0)) {
    _aevals = aevals;
    _bevals = bevals;
  }

  double call(const std::array<int, 4> &inds) const override {
    return 1.0 / ((*_aevals)(inds[0]) + (*_bevals)(inds[2]) - (*_aevals)(inds[1]) - (*_bevals)(inds[3]));
  }

  const einsums::Tensor<double, 1> *get_aevals() const { return _aevals; }

  void set_aevals(const einsums::Tensor<double, 1> *aevals) { _aevals = aevals; }

  const einsums::Tensor<double, 1> *get_bevals() const { return _bevals; }

  void set_bevals(const einsums::Tensor<double, 1> *bevals) { _bevals = bevals; }
};

class EinsumsUMP2 : public EinsumsUHF {
public:
  /// The constuctor
  EinsumsUMP2(std::shared_ptr<EinsumsUHF> ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsUMP2();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header() override;

  const einsums::TiledTensor<double, 4> &getTei() const { return tei_; }
  const einsums::TiledTensor<double, 4> &getTeiTransAA() const { return teitaa_; }
  const einsums::TiledTensor<double, 4> &getMP2AmpsAA() const {
    return MP2_ampsaa_;
  }
  const einsums::TiledTensor<double, 4> &getDenominatorAA() const {
    return denominatoraa_;
  }

  const einsums::TiledTensor<double, 4> &getTeiTransBB() const { return teitbb_; }
  const einsums::TiledTensor<double, 4> &getMP2AmpsBB() const {
    return MP2_ampsbb_;
  }
  const einsums::TiledTensor<double, 4> &getDenominatorBB() const {
    return denominatorbb_;
  }

  const einsums::TiledTensor<double, 4> &getTeiTransAB() const { return teitab_; }
  const einsums::TiledTensor<double, 4> &getMP2AmpsAB() const {
    return MP2_ampsab_;
  }
  const einsums::TiledTensor<double, 4> &getDenominatorAB() const {
    return denominatorab_;
  }

  einsums::TiledTensor<double, 4> &getTei() { return tei_; }
  einsums::TiledTensor<double, 4> &getTeiTransAA() { return teitaa_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsAA() { return MP2_ampsaa_; }
  einsums::TiledTensor<double, 4> &getDenominatorAA() { return denominatoraa_; }

  einsums::TiledTensor<double, 4> &getTeiTransBB() { return teitbb_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsBB() { return MP2_ampsbb_; }
  einsums::TiledTensor<double, 4> &getDenominatorBB() { return denominatorbb_; }

  einsums::TiledTensor<double, 4> &getTeiTransAB() { return teitab_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsAB() { return MP2_ampsab_; }
  einsums::TiledTensor<double, 4> &getDenominatorAB() { return denominatorab_; }

protected:
  /// The occupation per irrep.
  std::vector<int> aunocc_per_irrep_, bunocc_per_irrep_;
  // Offsets for the irreps.
  std::vector<int> irrep_offsets_;
  /// The two-electron integrals
  einsums::TiledTensor<double, 4> tei_;
  /// Transformed two-electron integrals
  einsums::TiledTensor<double, 4> teitaa_, teitbb_, teitab_;
  /// MP2 amplitudes.
  einsums::TiledTensor<double, 4> MP2_ampsaa_, MP2_ampsbb_, MP2_ampsab_;
  /// Function tensor for the MP2 denominator.
  einsums::TiledTensor<double, 4> denominatoraa_, denominatorbb_, denominatorab_;
  /// Sets up the integrals object
  void init_integrals();
  /// Sets up the spin integrals. true is alpha, false is beta.
  void setup_spin_integrals(bool spin1, bool spin2);
};

} // namespace einhf
} // namespace psi
