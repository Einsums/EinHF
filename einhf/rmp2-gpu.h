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
#include "rhf-gpu.h"
#include "rmp2.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

class GPUEinsumsRMP2 : public Wavefunction {
public:
  /// The constuctor
  GPUEinsumsRMP2(std::shared_ptr<GPUEinsumsRHF> ref_wfn, Options &options);
  /// The destuctor
  ~GPUEinsumsRMP2();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header();

  const einsums::BlockDeviceTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockDeviceTensor<double, 2> &getF() const { return F_; }
  const einsums::BlockDeviceTensor<double, 2> &getFt() const { return Ft_; }
  const einsums::BlockDeviceTensor<double, 2> &getC() const { return C_; }
  const einsums::BlockDeviceTensor<double, 2> &getCocc() const { return Cocc_; }
  const einsums::BlockDeviceTensor<double, 2> &getD() const { return D_; }
  const einsums::Tensor<double, 1> &getEvals() const { return evals_; }
  const einsums::TiledDeviceTensor<double, 4> &getTei() const { return tei_; }
  const einsums::TiledDeviceTensor<double, 4> &getTeiTrans() const { return teit_; }
  const einsums::TiledDeviceTensor<double, 4> &getMP2Amps() const {
    return MP2_amps_;
  }
  const RMP2ScaleTensor &getDenominator() const { return denominator_; }

  einsums::BlockDeviceTensor<double, 2> &getS() { return S_; }
  einsums::BlockDeviceTensor<double, 2> &getF() { return F_; }
  einsums::BlockDeviceTensor<double, 2> &getFt() { return Ft_; }
  einsums::BlockDeviceTensor<double, 2> &getC() { return C_; }
  einsums::BlockDeviceTensor<double, 2> &getCocc() { return Cocc_; }
  einsums::BlockDeviceTensor<double, 2> &getD() { return D_; }
  einsums::Tensor<double, 1> &getEvals() { return evals_; }
  einsums::TiledDeviceTensor<double, 4> &getTei() { return tei_; }
  einsums::TiledDeviceTensor<double, 4> &getTeiTrans() { return teit_; }
  einsums::TiledDeviceTensor<double, 4> &getMP2Amps() { return MP2_amps_; }
  RMP2ScaleTensor &getDenominator() { return denominator_; }

protected:
  /// The amount of information to print to the output file
  int print_;
  /// The number of doubly occupied orbitals
  int ndocc_;

  /// The occupation per irrep.
  std::vector<int> occ_per_irrep_, unocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_, irrep_offsets_;
  /// The names of the irreps.
  std::vector<std::string> irrep_names_;

  /// The number of symmetrized spin orbitals
  int nso_;
  /// The overlap matrix
  einsums::BlockDeviceTensor<double, 2> S_;
  /// The Fock Matrix
  einsums::BlockDeviceTensor<double, 2> F_;
  /// The transformed Fock matrix
  einsums::BlockDeviceTensor<double, 2> Ft_;
  /// The MO coefficients
  einsums::BlockDeviceTensor<double, 2> C_;
  /// The occupied MO coefficients
  einsums::BlockDeviceTensor<double, 2> Cocc_;
  /// The density matrix
  einsums::BlockDeviceTensor<double, 2> D_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evals_;
  /// The two-electron integrals
  einsums::TiledDeviceTensor<double, 4> tei_;
  /// Transformed two-electron integrals
  einsums::TiledDeviceTensor<double, 4> teit_;
  /// MP2 amplitudes.
  einsums::TiledDeviceTensor<double, 4> MP2_amps_;
  /// Function tensor for the MP2 denominator.
  RMP2ScaleTensor denominator_;
  /// Sets up the integrals object
  void init_integrals();

  void set_tile(const einsums::TiledDeviceTensor<double, 4> &temp, int i, int a,
                int j, int b);
};

} // namespace einhf
} // namespace psi
