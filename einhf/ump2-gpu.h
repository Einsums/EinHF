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
#include "ump2.h"
#include "uhf-gpu.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

class GPUEinsumsUMP2 : public Wavefunction {
public:
  /// The constuctor
  GPUEinsumsUMP2(std::shared_ptr<GPUEinsumsUHF> ref_wfn, Options &options);
  /// The destuctor
  ~GPUEinsumsUMP2();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy() override;

  virtual void print_header();

  const einsums::BlockDeviceTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockDeviceTensor<double, 2> &getFa() const { return Fa_; }
  const einsums::BlockDeviceTensor<double, 2> &getFta() const { return Fta_; }
  const einsums::BlockDeviceTensor<double, 2> &getCa() const { return Ca_; }
  const einsums::BlockDeviceTensor<double, 2> &getCocca() const { return Cocca_; }
  const einsums::BlockDeviceTensor<double, 2> &getDa() const { return Da_; }
  const einsums::Tensor<double, 1> &getEvalsa() const { return evalsa_; }
  const einsums::BlockDeviceTensor<double, 2> &getFb() const { return Fa_; }
  const einsums::BlockDeviceTensor<double, 2> &getFtb() const { return Ftb_; }
  const einsums::BlockDeviceTensor<double, 2> &getCb() const { return Cb_; }
  const einsums::BlockDeviceTensor<double, 2> &getCoccb() const { return Coccb_; }
  const einsums::BlockDeviceTensor<double, 2> &getDb() const { return Db_; }
  const einsums::Tensor<double, 1> &getEvalsb() const { return evalsb_; }

  const einsums::TiledTensor<double, 4> &getTeiTransAA() const {
    return teitaa_;
  }
  const einsums::TiledTensor<double, 4> &getMP2AmpsAA() const {
    return MP2_ampsaa_;
  }
  const UMP2ScaleTensor &getDenominatorAA() const { return denominatoraa_; }

  const einsums::TiledTensor<double, 4> &getTeiTransBB() const {
    return teitbb_;
  }
  const einsums::TiledTensor<double, 4> &getMP2AmpsBB() const {
    return MP2_ampsbb_;
  }
  const UMP2ScaleTensor &getDenominatorBB() const { return denominatorbb_; }

  const einsums::TiledTensor<double, 4> &getTeiTransAB() const {
    return teitab_;
  }
  const einsums::TiledTensor<double, 4> &getMP2AmpsAB() const {
    return MP2_ampsab_;
  }
  const UMP2ScaleTensor &getDenominatorAB() const { return denominatorab_; }

  einsums::BlockDeviceTensor<double, 2> &getS() { return S_; }
  einsums::BlockDeviceTensor<double, 2> &getFa() { return Fa_; }
  einsums::BlockDeviceTensor<double, 2> &getFta() { return Fta_; }
  einsums::BlockDeviceTensor<double, 2> &getCa() { return Ca_; }
  einsums::BlockDeviceTensor<double, 2> &getCocca() { return Cocca_; }
  einsums::BlockDeviceTensor<double, 2> &getDa() { return Da_; }
  einsums::Tensor<double, 1> &getEvalsa() { return evalsa_; }
  einsums::BlockDeviceTensor<double, 2> &getFb() { return Fb_; }
  einsums::BlockDeviceTensor<double, 2> &getFtb() { return Ftb_; }
  einsums::BlockDeviceTensor<double, 2> &getCb() { return Cb_; }
  einsums::BlockDeviceTensor<double, 2> &getCoccb() { return Coccb_; }
  einsums::BlockDeviceTensor<double, 2> &getDb() { return Db_; }
  einsums::Tensor<double, 1> &getEvalsb() { return evalsb_; }

  einsums::TiledTensor<double, 4> &getTeiTransAA() { return teitaa_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsAA() { return MP2_ampsaa_; }
  UMP2ScaleTensor &getDenominatorAA() { return denominatoraa_; }

  einsums::TiledTensor<double, 4> &getTeiTransBB() { return teitbb_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsBB() { return MP2_ampsbb_; }
  UMP2ScaleTensor &getDenominatorBB() { return denominatorbb_; }

  einsums::TiledTensor<double, 4> &getTeiTransAB() { return teitab_; }
  einsums::TiledTensor<double, 4> &getMP2AmpsAB() { return MP2_ampsab_; }
  UMP2ScaleTensor &getDenominatorAB() { return denominatorab_; }

protected:
  /// The amount of information to print to the output file
  int print_;
  /// The number of doubly occupied orbitals
  int naocc_, nbocc_;

  /// The occupation per irrep.
  std::vector<int> aocc_per_irrep_, aunocc_per_irrep_;
  std::vector<int> bocc_per_irrep_, bunocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_, irrep_offsets_;
  /// The names of the irreps.
  std::vector<std::string> irrep_names_;

  /// The number of symmetrized spin orbitals
  int nso_;
  /// The overlap matrix
  einsums::BlockDeviceTensor<double, 2> S_;
  /// The Fock Matrix
  einsums::BlockDeviceTensor<double, 2> Fa_;
  /// The transformed Fock matrix
  einsums::BlockDeviceTensor<double, 2> Fta_;
  /// The MO coefficients
  einsums::BlockDeviceTensor<double, 2> Ca_;
  /// The occupied MO coefficients
  einsums::BlockDeviceTensor<double, 2> Cocca_;
  /// The density matrix
  einsums::BlockDeviceTensor<double, 2> Da_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evalsa_;
  /// The Fock Matrix
  einsums::BlockDeviceTensor<double, 2> Fb_;
  /// The transformed Fock matrix
  einsums::BlockDeviceTensor<double, 2> Ftb_;
  /// The MO coefficients
  einsums::BlockDeviceTensor<double, 2> Cb_;
  /// The occupied MO coefficients
  einsums::BlockDeviceTensor<double, 2> Coccb_;
  /// The density matrix
  einsums::BlockDeviceTensor<double, 2> Db_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evalsb_;
  /// The two-electron integrals
  einsums::TiledTensor<double, 4> tei_;
  /// Transformed two-electron integrals
  einsums::TiledTensor<double, 4> teitaa_, teitbb_, teitab_;
  /// MP2 amplitudes.
  einsums::TiledTensor<double, 4> MP2_ampsaa_, MP2_ampsbb_, MP2_ampsab_;
  /// Function tensor for the MP2 denominator.
  UMP2ScaleTensor denominatoraa_, denominatorbb_, denominatorab_;
  /// Sets up the integrals object
  void init_integrals();
  void set_tile(const einsums::TiledTensor<double, 4> &temp,
                einsums::TiledTensor<double, 4> &teit,
                UMP2ScaleTensor &denominator, einsums::TiledTensor<double, 4> &MP2_amps,
                const std::vector<int> &aocc_per_irrep,
                const std::vector<int> &bocc_per_irrep, int i, int a, int j,
                int b);
};

} // namespace einhf
} // namespace psi
