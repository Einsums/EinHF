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

struct MP2ScaleFunction
    : public virtual einsums::tensor_props::FunctionTensorBase<double, 4>,
      virtual einsums::tensor_props::CoreTensorBase {
private:
  const einsums::Tensor<double, 1> *_evals;

public:
  MP2ScaleFunction() = default;
  MP2ScaleFunction(const MP2ScaleFunction &) = default;

  MP2ScaleFunction(std::string name, const einsums::Tensor<double, 1> *evals)
      : einsums::tensor_props::FunctionTensorBase<double, 4>(
            name, evals->dim(0), evals->dim(0), evals->dim(0), evals->dim(0)) {
    _evals = evals;
  }

  double call(const std::array<int, 4> &inds) const override {
    return 1.0 / ((*_evals)(inds[0]) + (*_evals)(inds[2]) - (*_evals)(inds[1]) - (*_evals)(inds[3]));
  }

  const einsums::Tensor<double, 1> *get_evals() const { return _evals; }

  void set_evals(const einsums::Tensor<double, 1> *evals) { _evals = evals; }
};

class EinsumsRMP2 : public EinsumsRHF {
public:
  /// The constuctor
  EinsumsRMP2(std::shared_ptr<EinsumsRHF> ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsRMP2();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header() override;

  const einsums::TiledTensor<double, 4> &getTei() const { return tei_; }
  const einsums::TiledTensor<double, 4> &getTeiTrans() const { return teit_; }
  const einsums::TiledTensor<double, 4> &getMP2Amps() const {
    return MP2_amps_;
  }
  const einsums::TiledTensor<double, 4> &getDenominator() const {
    return denominator_;
  }

  einsums::TiledTensor<double, 4> &getTei() { return tei_; }
  einsums::TiledTensor<double, 4> &getTeiTrans() { return teit_; }
  einsums::TiledTensor<double, 4> &getMP2Amps() { return MP2_amps_; }
  einsums::TiledTensor<double, 4> &getDenominator() { return denominator_; }

protected:
  /// The occupation per irrep.
  std::vector<int> unocc_per_irrep_;
  // Offsets for the irreps.
  std::vector<int> irrep_offsets_;
  /// The two-electron integrals
  einsums::TiledTensor<double, 4> tei_;
  /// Transformed two-electron integrals
  einsums::TiledTensor<double, 4> teit_;
  /// MP2 amplitudes.
  einsums::TiledTensor<double, 4> MP2_amps_;
  /// Function tensor for the MP2 denominator.
  einsums::TiledTensor<double, 4> denominator_;
  /// Sets up the integrals object
  void init_integrals();
};

} // namespace einhf
} // namespace psi
