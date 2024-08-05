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
    return std::apply(*_evals, inds);
  }

  const einsums::Tensor<double, 1> *get_evals() const { return _evals; }

  void set_evals(const einsums::Tensor<double, 1> *evals) { _evals = evals; }
};

class EinsumsRMP2 : public Wavefunction {
public:
  /// The constuctor
  EinsumsRMP2(std::shared_ptr<EinsumsSCF> ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsRMP2();
  /// Computes the SCF energy, and returns it
  double compute_energy();

  void print_header();

  const einsums::BlockTensor<double, 2> &getH() const { return H_; }
  const einsums::BlockTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockTensor<double, 2> &getX() const { return X_; }
  const einsums::BlockTensor<double, 2> &getF() const { return F_; }
  const einsums::BlockTensor<double, 2> &getFt() const { return Ft_; }
  const einsums::BlockTensor<double, 2> &getC() const { return C_; }
  const einsums::BlockTensor<double, 2> &getCocc() const { return Cocc_; }
  const einsums::BlockTensor<double, 2> &getD() const { return D_; }
  const einsums::Tensor<double, 1> &getEvals() const { return evals_; }

  einsums::BlockTensor<double, 2> &getH() { return H_; }
  einsums::BlockTensor<double, 2> &getS() { return S_; }
  einsums::BlockTensor<double, 2> &getX() { return X_; }
  einsums::BlockTensor<double, 2> &getF() { return F_; }
  einsums::BlockTensor<double, 2> &getFt() { return Ft_; }
  einsums::BlockTensor<double, 2> &getC() { return C_; }
  einsums::BlockTensor<double, 2> &getCocc() { return Cocc_; }
  einsums::BlockTensor<double, 2> &getD() { return D_; }
  einsums::Tensor<double, 1> &getEvals() { return evals_; }

protected:
  /// The amount of information to print to the output file
  int print_;
  /// The number of doubly occupied orbitals
  int ndocc_;

  /// The occupation per irrep.
  std::vector<int> occ_per_irrep_, unocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_;
  // Offsets for the irreps.
  std::vector<int> irrep_offsets_;

  /// The number of symmetrized spin orbitals
  int nso_;
  /// The maximum number of iterations
  int maxiter_;
  /// The number of DIIS iterations to hold.
  int diis_max_iters_;
  /// The nuclear repulsion energy
  double e_nuc_;
  /// The convergence criterion for the density
  double d_convergence_;
  /// The convergence criterion for the energy
  double e_convergence_;
  /// The one electron integrals
  einsums::BlockTensor<double, 2> H_;
  /// The overlap matrix
  einsums::BlockTensor<double, 2> S_;
  /// The inverse square root of the overlap matrix
  einsums::BlockTensor<double, 2> X_;
  /// The Fock Matrix
  einsums::BlockTensor<double, 2> F_;
  /// The transformed Fock matrix
  einsums::BlockTensor<double, 2> Ft_;
  /// The MO coefficients
  einsums::BlockTensor<double, 2> C_;
  /// The occupied MO coefficients
  einsums::BlockTensor<double, 2> Cocc_;
  /// The density matrix
  einsums::BlockTensor<double, 2> D_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evals_;
  /// The two-electron integrals
  einsums::TiledTensor<double, 4> tei_;
  /// Transformed two-electron integrals
  einsums::TiledTensor<double, 4> teit_;
  ///
  einsums::TiledTensor<double, 4> MP2_amps_;
  /// Function tensor for the MP2 denominator.
  einsums::TiledTensor<double, 4> denominator_;
  /// Sets up the integrals object
  void init_integrals();
};

} // namespace einhf
} // namespace psi
