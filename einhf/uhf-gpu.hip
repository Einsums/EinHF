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

#include "uhf-gpu.h"

#include "einsums.hpp"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include <LinearAlgebra.hpp>
#include <_Common.hpp>
#include <_Index.hpp>

static std::string to_lower(const std::string &str) {
  std::string out(str);
  std::transform(str.begin(), str.end(), out.begin(),
                 [](char c) { return std::tolower(c); });
  return out;
}

namespace psi {
namespace einhf {

GPUEinsumsUHF::GPUEinsumsUHF(SharedWavefunction ref_wfn, Options &options)
    : Wavefunction(options) {

  timer_on("Setup Wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  print_ = options_.get_int("PRINT");
  print_header();

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();

  maxiter_ = options_.get_int("SCF_MAXITER");
  e_convergence_ = options_.get_double("E_CONVERGENCE");
  d_convergence_ = options_.get_double("D_CONVERGENCE");
  if (options_.get_bool("DIIS")) {
    diis_max_iters_ = options_.get_int("DIIS_MAX_VECS");
  } else {
    diis_max_iters_ = 0;
    outfile->Printf("Turning DIIS off.\n");
  }

  init_integrals();

  // Set Wavefunction matrices
  X_.set_name("S^1/2");
  Fa_.set_name("Alpha Fock Matrix");
  Fta_.set_name("Transformed Alpha Fock Matrix");
  Ca_.set_name("Alpha MO Coefficients");
  Cocca_.set_name("Occupied Alpha MO Coefficients");
  Da_.set_name("Alpha Density Matrix");

  Fb_.set_name("Beta Fock Matrix");
  Ftb_.set_name("Transformed Beta Fock Matrix");
  Cb_.set_name("Beta MO Coefficients");
  Coccb_.set_name("Occupied Beta MO Coefficients");
  Db_.set_name("Beta Density Matrix");

  for (int i = 0; i < this->nirrep_; i++) {
    X_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Fa_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
    Fb_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Fta_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
    Ftb_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Ca_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
    Cb_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Cocca_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
    Coccb_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Da_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
    Db_.push_block(einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        irrep_sizes_[i], irrep_sizes_[i]));
  }
  timer_off("Setup Wavefunction");
}

GPUEinsumsUHF::~GPUEinsumsUHF() {}

void GPUEinsumsUHF::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  // Determine the number of electrons in the system
  // The molecule object is built into all wavefunctions
  int charge = molecule_->molecular_charge();
  int nelec = 0;
  for (int i = 0; i < molecule_->natom(); ++i) {
    nelec += (int)molecule_->Z(i);
  }
  nelec -= charge;
  nbocc_ = (nelec - molecule_->multiplicity() + 1) / 2;
  naocc_ = nbocc_ + molecule_->multiplicity() - 1;

  outfile->Printf("    There are %d alpha occupied orbitals and %d beta "
                  "occupied orbitals.\n",
                  naocc_, nbocc_);
  molecule_->print();
  if (print_ > 1) {
    basisset_->print_detail();
    options_.print();
  }

  // nso_ = basisset_->nso();

  // Nuclear repulsion without a field
  e_nuc_ = molecule_->nuclear_repulsion_energy({0, 0, 0});
  outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", e_nuc_);

  // Make a MintsHelper object to help
  auto mints = std::make_shared<MintsHelper>(basisset_);

  // These don't need to be declared, because they belong to the class
  auto S_mat = mints->so_overlap();

  // Core Hamiltonian is Kinetic + Potential
  auto H_mat = mints->so_kinetic();

  this->nirrep_ = (int)S_mat->nirrep();

  aocc_per_irrep_.resize(nirrep_);
  bocc_per_irrep_.resize(nirrep_);

  H_mat->add(mints->so_potential());

  H_.set_name("Hamiltonian");
  S_.set_name("Overlap");

  for (int i = 0; i < S_mat->nirrep(); i++) {
    irrep_sizes_.push_back(S_mat->rowdim(i));
    auto S_block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        S_mat->rowdim(i), S_mat->coldim(i));

    S_block.read(S_mat->get_pointer(i));

    S_.push_block(S_block);
  }

  for (int i = 0; i < S_mat->nirrep(); i++) {
    auto H_block = einsums::DeviceTensor<double, 2>(
        molecule_->irrep_labels().at(i), einsums::detail::DEV_ONLY,
        H_mat->rowdim(i), H_mat->coldim(i));

    H_block.read(H_mat->get_pointer(i));

    H_.push_block(H_block);
  }

  if (print_ > 3) {
    fprintln(*outfile->stream(), S_);
    fprintln(*outfile->stream(), H_);
    outfile->stream()->flush();
  }

  outfile->Printf("    Forming JK object\n\n");

  size_t total_memory = Process::environment.get_memory() / 8 *
                        options_.get_double("SCF_MEM_SAFETY_FACTOR");

  // Construct a JK object that compute J and K SCF matrices very efficiently
  jk_ = JK::build_JK(basisset_, mintshelper_->get_basisset("DF_BASIS_SCF"),
                     options_, false, total_memory);

  size_t jk_size = jk_->memory_estimate();

  if (jk_size < total_memory) {
    jk_->set_memory(jk_size);
  } else {
    jk_->set_memory(total_memory * 0.9);
  }

  jk_->initialize();
  jk_->print_header();
}

double GPUEinsumsUHF::compute_electronic_energy(
    const einsums::BlockDeviceTensor<double, 2> &F,
    const einsums::BlockDeviceTensor<double, 2> &D) {
  // Compute the electronic energy: (H + F)_pq * D_pq -> energy

  einsums::DeviceTensor<double, 0> e_tens;
  auto temp = einsums::BlockDeviceTensor<double, 2>(
      "temp", einsums::detail::DEV_ONLY, H_.vector_dims());

  temp = H_;
  temp += F;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &e_tens, 1.0,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      D,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      temp);
  return (double)e_tens;
}

void GPUEinsumsUHF::update_Cocc(
    const einsums::DeviceTensor<double, 1> &alpha_energies,
    const einsums::DeviceTensor<double, 1> &beta_energies) {
  // Update occupation.

  for (int i = 0; i < nirrep_; i++) {
    aocc_per_irrep_[i] = 0;
    bocc_per_irrep_[i] = 0;
  }

  // Alpha
  for (int i = 0; i < naocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (aocc_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = alpha_energies(S_.block_range(j)[0] + aocc_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    aocc_per_irrep_[irrep_occ]++;
  }

  // Beta
  for (int i = 0; i < nbocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (bocc_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = beta_energies(S_.block_range(j)[0] + bocc_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    bocc_per_irrep_[irrep_occ]++;
  }

  Cocca_.zero();
  Coccb_.zero();

// Alpha
#pragma omp parallel for
  for (int i = 0; i < nirrep_; i++) {
    Cocca_[i](einsums::AllT{}, einsums::Range(0, aocc_per_irrep_[i])) =
        Ca_[i](einsums::AllT{}, einsums::Range(0, aocc_per_irrep_[i]));
  }
// Beta
#pragma omp parallel for
  for (int i = 0; i < nirrep_; i++) {
    Coccb_[i](einsums::AllT{}, einsums::Range(0, bocc_per_irrep_[i])) =
        Cb_[i](einsums::AllT{}, einsums::Range(0, bocc_per_irrep_[i]));
  }
}

void GPUEinsumsUHF::compute_diis_coefs(
    const std::deque<einsums::BlockDeviceTensor<double, 2>> &errors,
    std::vector<double> *out) const {
  einsums::Tensor<double, 2> *B_mat = new einsums::Tensor<double, 2>(
      "DIIS error matrix", errors.size() + 1, errors.size() + 1);

  B_mat->zero();
  (*B_mat)(einsums::Range{errors.size(), errors.size() + 1},
           einsums::Range{0, errors.size()}) = 1.0;
  (*B_mat)(einsums::Range{0, errors.size()},
           einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
  for (int i = 0; i < errors.size(); i++) {
#pragma omp parallel for
    for (int j = 0; j <= i; j++) {
      (*B_mat)(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
      (*B_mat)(j, i) = (*B_mat)(i, j);
    }
  }

  einsums::Tensor<double, 2> res_mat =
      einsums::Tensor<double, 2>("DIIS result matrix", 1, errors.size() + 1);

  res_mat.zero();
  res_mat(0, errors.size()) = 1.0;

  einsums::linear_algebra::gesv(B_mat, &res_mat);

  delete B_mat;

  out->resize(errors.size());

  for (int i = 0; i < errors.size(); i++) {
    out->at(i) = res_mat(0, i);
  }
}

void GPUEinsumsUHF::compute_diis_fock(
    const std::vector<double> &coefs,
    const std::deque<einsums::BlockDeviceTensor<double, 2>> &focks,
    einsums::BlockDeviceTensor<double, 2> *out) const {

  out->zero();

  for (int i = 0; i < coefs.size(); i++) {
    einsums::linear_algebra::axpy(coefs[i], focks[i], out);
  }
}

double GPUEinsumsUHF::compute_energy() {
  if (diis_max_iters_ != 0) {
    outfile->Printf("Performing DIIS with %d vectors.\n", diis_max_iters_);
  } else {
    outfile->Printf("Turning DIIS off.\n");
  }

  // Allocate a few temporary matrices
  auto Temp1a = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha Temporary Array 1", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Temp2a = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha Temporary Array 2", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Temp1b = new einsums::BlockDeviceTensor<double, 2>(
      "Beta Temporary Array 1", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Temp2b = new einsums::BlockDeviceTensor<double, 2>(
      "Beta Temporary Array 2", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto FDSa = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha FDS", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto SDFa = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha SDF", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto FDSb = new einsums::BlockDeviceTensor<double, 2>(
      "Beta FDS", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto SDFb = new einsums::BlockDeviceTensor<double, 2>(
      "Beta SDF", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Evecsa = new einsums::BlockDeviceTensor<double, 2>(
      "AlphaEigenvectors", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Evalsa = new einsums::DeviceTensor<double, 1>(
      "AlphaEigenvalues", einsums::detail::DEV_ONLY, nso_);
  auto Evecsb = new einsums::BlockDeviceTensor<double, 2>(
      "BetaEigenvectors", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Evalsb = new einsums::DeviceTensor<double, 1>(
      "BetaEigenvalues", einsums::detail::DEV_ONLY, nso_);

  auto Ja = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha J matrix", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Ka = new einsums::BlockDeviceTensor<double, 2>(
      "Alpha K matrix", einsums::detail::DEV_ONLY, irrep_sizes_);

  auto Jb = new einsums::BlockDeviceTensor<double, 2>(
      "Beta J matrix", einsums::detail::DEV_ONLY, irrep_sizes_);
  auto Kb = new einsums::BlockDeviceTensor<double, 2>(
      "Beta K matrix", einsums::detail::DEV_ONLY, irrep_sizes_);

  std::deque<einsums::BlockDeviceTensor<double, 2>>
      *errorsa = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0),
      *errorsb = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0),
      *focksa = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0),
      *focksb = new std::deque<einsums::BlockDeviceTensor<double, 2>>(0);
  std::vector<double> *coefsa = new std::vector<double>(0),
                      *coefsb = new std::vector<double>(0),
                      *error_valsa = new std::vector<double>(0),
                      *error_valsb = new std::vector<double>(0);

  std::vector<int> old_occsa, old_occsb;

  einsums::DeviceTensor<double, 0> *dRMS_tensa =
                                       new einsums::DeviceTensor<double, 0>(),
                                   *dRMS_tensb =
                                       new einsums::DeviceTensor<double, 0>();
  double *elec_a = new double(), *elec_b = new double();

// Form the X_ matrix (S^-1/2)
#pragma omp task depend(in : this->S_) depend(out : this -> X_)
  {
    timer_on("Form X");
    X_ = einsums::linear_algebra::pow(S_, -0.5);
    timer_off("Form X");
  }

#pragma omp task depend(in : this->H_) depend(out : this -> Fa_)
  {
    timer_on("Form Fa");
    Fa_ = H_;
    timer_off("Form Fa");
  }

#pragma omp task depend(in : this->H_) depend(out : this -> Fb_)
  {
    timer_on("Form Fb");
    Fb_ = H_;
    timer_off("Form Fb");
  }

// Alpha
#pragma omp task depend(in : this->Fa_, this->X_)                              \
    depend(out : *Temp1a, this -> Fta_, this->Ca_, *Evalsa, *Evecsa)
  {
    timer_on("Form Ca");
    einsums::linear_algebra::gemm<false, false>(1.0, Fa_, X_, 0.0, Temp1a);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1a, 0.0, &Fta_);

    *Evecsa = Fta_;

    einsums::linear_algebra::syev(Evecsa, Evalsa);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsa, 0.0, &Ca_);
    timer_off("Form Ca");
  }

// Beta
#pragma omp task depend(in : this->Fb_, this->X_)                              \
    depend(out : *Temp1b, this -> Ftb_, this->Cb_, *Evalsb, *Evecsb)
  {
    timer_on("Form Cb");
    einsums::linear_algebra::gemm<false, false>(1.0, Fb_, X_, 0.0, Temp1b);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1b, 0.0, &Ftb_);

    *Evecsb = Ftb_;

    einsums::linear_algebra::syev(Evecsb, Evalsb);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsb, 0.0, &Cb_);
    timer_off("Form Cb");
  }
#pragma omp taskwait depend(in : *Evalsa, *Evalsb, this->Ca_, this->Cb_)       \
    depend(out : this -> Cocca_, this->Coccb_)

  // Update Cocc.
  update_Cocc(*Evalsa, *Evalsb);

#pragma omp task depend(in : this->Cocca_) depend(out : this -> Da_)
  {
    timer_on("Form Da");
    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &Da_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Cocca_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Cocca_);
    timer_off("Form Da");
  }

#pragma omp task depend(in : this->Coccb_) depend(out : this -> Db_)
  {
    timer_on("Form Db");
    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &Db_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Coccb_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Coccb_);
    timer_off("Form Db");
  }

  if (print_ > 3) {
#pragma omp taskwait
    outfile->Printf(
        "MO Coefficients and density from Core Hamiltonian guess:\n");
    fprintln(*outfile->stream(), X_);
    fprintln(*outfile->stream(), Ca_);
    fprintln(*outfile->stream(), Da_);
    fprintln(*outfile->stream(), *Evalsa);
    fprintln(*outfile->stream(), Cocca_);
    fprintln(*outfile->stream(), Cb_);
    fprintln(*outfile->stream(), Db_);
    fprintln(*outfile->stream(), *Evalsb);
    fprintln(*outfile->stream(), Coccb_);
  }

  int iter = 1;
  bool converged = false;
  double e_old;
  double e_new = e_nuc_;

  timer_on("Compute electronic energy");
#pragma omp task depend(in : this->H_, this->Fa_, this->Da_)                   \
    depend(out : *elec_a)
  { *elec_a = compute_electronic_energy(Fa_, Da_); }
#pragma omp task depend(in : this->H_, this->Fb_, this->Db_)                   \
    depend(out : *elec_b)
  { *elec_b = compute_electronic_energy(Fb_, Db_); }
#pragma omp taskwait depend(in : *elec_a, *elec_b)

  e_new += (*elec_a + *elec_b) / 2;
  timer_off("Compute electronic energy");

  outfile->Printf("    Energy from core Hamiltonian guess: %20.16f\n\n", e_new);

  old_occsa = aocc_per_irrep_;
  old_occsb = bocc_per_irrep_;

  outfile->Printf("    Initial Occupation:\n         \t");

  for (int i = 0; i < S_.num_blocks(); i++) {
    outfile->Printf("%4s\t", S_.name(i).c_str());
  }

  outfile->Printf("\n    AOCC \t");
  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", aocc_per_irrep_[i]);
  }

  outfile->Printf("\n    BOCC \t");
  for (int i = 0; i < bocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", bocc_per_irrep_[i]);
  }

  outfile->Printf("\n    VIRT \t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i] - aocc_per_irrep_[i]);
  }

  outfile->Printf("\n    Total\t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i]);
  }

  outfile->Printf("\n");

  outfile->Printf(
      "    *===========================================================*\n");
  outfile->Printf(
      "    * Iter       Energy            delta E    ||gradient||      *\n");
  outfile->Printf(
      "    *-----------------------------------------------------------*\n");

  while (!converged && iter < maxiter_) {
    e_old = e_new;

// Add the core Hamiltonian term to the Fock operator
#pragma omp task depend(in : this->H_) depend(out : this -> Fa_)
    {
      timer_on("Form Fa");
      Fa_ = H_;
    }

#pragma omp task depend(in : this->H_) depend(out : this -> Fb_)
    {
      timer_on("Form Fb");
      Fb_ = H_;
    }

    // The JK object handles all of the two electron integrals
    // To enhance efficiency it does use the density, but the orbitals
    // themselves D_uv = C_ui C_vj J_uv = I_uvrs D_rs K_uv = I_urvs D_rs

    // Here we clear the old Cocc and push_back our new one
    std::vector<SharedMatrix> &Cl = jk_->C_left();
    Cl.clear();

    SharedMatrix CTempa = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                   aocc_per_irrep_.data());
    SharedMatrix CTempb = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                   bocc_per_irrep_.data());
// Alpha
#pragma omp parallel for
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
#pragma omp parallel for
      for (int j = 0; j < irrep_sizes_[i]; j++) {
#pragma omp parallel for
        for (int k = 0; k < aocc_per_irrep_[i]; k++) {
          (*CTempa.get())(i, j, k) = Cocca_[i](j, k);
        }
      }
    }

// Beta
#pragma omp parallel for
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
#pragma omp parallel for
      for (int j = 0; j < irrep_sizes_[i]; j++) {
#pragma omp parallel for
        for (int k = 0; k < bocc_per_irrep_[i]; k++) {
          (*CTempb.get())(i, j, k) = Coccb_[i](j, k);
        }
      }
    }

    Cl.push_back(CTempa);
    Cl.push_back(CTempb);
    jk_->compute();

    // Obtain the new J and K matrices
    const std::vector<SharedMatrix> &J_mat = jk_->J();
    const std::vector<SharedMatrix> &K_mat = jk_->K();

    // Proceed as normal

#pragma omp task depend(out : *Ja)
    {
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        Ja->block(i).read(J_mat[0]->get_pointer(i));
      }
    }

#pragma omp task depend(out : *Jb)
    {
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        Jb->block(i).read(J_mat[1]->get_pointer(i));
      }
    }

#pragma omp task depend(out : *Ka)
    {
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        Ka->block(i).read(K_mat[0]->get_pointer(i));
      }
    }

#pragma omp task depend(out : *Kb)
    {
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        Kb->block(i).read(K_mat[1]->get_pointer(i));
      }
    }

#pragma omp task depend(in : *Ja, *Jb, *Ka) depend(out : this -> Fa_)
    {
      Fa_ += *Ja;
      Fa_ += *Jb;
      Fa_ -= *Ka;

      timer_off("Form Fa");
    }

#pragma omp task depend(in : *Ja, *Jb, *Kb) depend(out : this -> Fb_)
    {
      Fb_ += *Ja;
      Fb_ += *Jb;
      Fb_ -= *Kb;

      timer_off("Form Fb");
    }

#pragma omp task depend(in : this->Da_, this->S_, this->Fa_)                   \
    depend(out : *Temp1a, *FDSa)
    {

      // Compute the orbital gradient, FDS-SDF
      einsums::linear_algebra::gemm<false, false>(1.0, Da_, S_, 0.0, Temp1a);
      einsums::linear_algebra::gemm<false, false>(1.0, Fa_, *Temp1a, 0.0, FDSa);
    }

#pragma omp task depend(in : this->Da_, this->S_, this->Fa_)                   \
    depend(out : *Temp2a, *SDFa)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, Da_, Fa_, 0.0, Temp2a);
      einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp2a, 0.0, SDFa);
    }

#pragma omp task depend(in : this->Db_, this->S_, this->Fb_)                   \
    depend(out : *Temp1b, *FDSb)
    {

      // Compute the orbital gradient, FDS-SDF
      einsums::linear_algebra::gemm<false, false>(1.0, Db_, S_, 0.0, Temp1b);
      einsums::linear_algebra::gemm<false, false>(1.0, Fb_, *Temp1b, 0.0, FDSb);
    }

#pragma omp task depend(in : this->Db_, this->S_, this->Fb_)                   \
    depend(out : *Temp2b, *SDFb)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, Db_, Fb_, 0.0, Temp2b);
      einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp2b, 0.0, SDFb);
    }

#pragma omp task depend(in : *FDSa, *SDFa) depend(out : *Temp1a)
    {
      *Temp1a = *FDSa;
      *Temp1a -= *SDFa;
    }

#pragma omp task depend(in : *FDSb, *SDFb) depend(out : *Temp1b)
    {
      *Temp1b = *FDSb;
      *Temp1b -= *SDFb;
    }

    if (diis_max_iters_ > 0) {
#pragma omp task depend(in : *Temp1a)                                          \
    depend(inout : this -> Fa_, *error_valsa)                                  \
    depend(out : *errorsa, *focksa, *coefsa)
      {

        if (errorsa->size() == diis_max_iters_) {
          double max_error = -INFINITY;
          int max_ind = -1;

          for (int i = 0; i < diis_max_iters_; i++) {
            if (error_valsa->at(i) > max_error) {
              max_error = error_valsa->at(i);
              max_ind = i;
            }
          }

          focksa->at(max_ind) = Fa_;
          errorsa->at(max_ind) = *Temp1a;
          error_valsa->at(max_ind) =
              einsums::linear_algebra::dot(*Temp1a, *Temp1a);
        } else {
          errorsa->push_back(*Temp1a);
          focksa->push_back(Fa_);
          error_valsa->push_back(
              einsums::linear_algebra::dot(*Temp1a, *Temp1a));
        }

        compute_diis_coefs(*errorsa, coefsa);

        compute_diis_fock(*coefsa, *focksa, &Fa_);
      }
#pragma omp task depend(in : *Temp1b)                                          \
    depend(inout : this -> Fb_, *error_valsb)                                  \
    depend(out : *errorsb, *focksb, *coefsb)
      {

        if (errorsb->size() == diis_max_iters_) {
          double max_error = -INFINITY;
          int max_ind = -1;

          for (int i = 0; i < diis_max_iters_; i++) {
            if (error_valsb->at(i) > max_error) {
              max_error = error_valsb->at(i);
              max_ind = i;
            }
          }

          focksb->at(max_ind) = Fb_;
          errorsb->at(max_ind) = *Temp1b;
          error_valsb->at(max_ind) =
              einsums::linear_algebra::dot(*Temp1b, *Temp1b);
        } else {
          errorsb->push_back(*Temp1b);
          focksb->push_back(Fb_);
          error_valsb->push_back(
              einsums::linear_algebra::dot(*Temp1b, *Temp1b));
        }

        compute_diis_coefs(*errorsb, coefsb);

        compute_diis_fock(*coefsb, *focksb, &Fb_);
      }
    }

    // Density RMS
    *dRMS_tensa = 0;
    *dRMS_tensb = 0;

#pragma omp task depend(in : *Temp1a) depend(out : *dRMS_tensa)
    {
      einsums::tensor_algebra::einsum(
          0.0, einsums::tensor_algebra::Indices{}, dRMS_tensa,
          1.0 / (nso_ * nso_),
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1a,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1a);
    }

#pragma omp task depend(in : *Temp1b) depend(out : *dRMS_tensb)
    {
      einsums::tensor_algebra::einsum(
          0.0, einsums::tensor_algebra::Indices{}, dRMS_tensb,
          1.0 / (nso_ * nso_),
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1b,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1b);
    }

#pragma omp taskwait depend(in : *dRMS_tensa, *dRMS_tensb)
    double dRMS = std::sqrt((double)*dRMS_tensa + (double)*dRMS_tensb);

    // Compute the energy
    timer_on("Compute electronic energy");
#pragma omp task depend(in : this->H_, this->Fa_, this->Da_)                   \
    depend(out : *elec_a)
    { *elec_a = compute_electronic_energy(Fa_, Da_); }
#pragma omp task depend(in : this->H_, this->Fb_, this->Db_)                   \
    depend(out : *elec_b)
    { *elec_b = compute_electronic_energy(Fb_, Db_); }
#pragma omp taskwait depend(in : *elec_a, *elec_b)

    e_new = e_nuc_;

    e_new += (*elec_a + *elec_b) / 2;
    timer_off("Compute electronic energy");
    double dE = e_new - e_old;

    converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

    outfile->Printf("    * %3d %20.14f    %9.2e    %9.2e    ", iter, e_new, dE,
                    dRMS);
    if (focksa->size() > 0) {
      outfile->Printf("DIIS*\n");
    } else {
      outfile->Printf("    *\n");
    }

    // Alpha
#pragma omp task depend(in : this->Fa_, this->X_)                              \
    depend(out : *Temp1a, this -> Fta_, this->Ca_, *Evalsa, *Evecsa)
    {
      timer_on("Form Ca");
      einsums::linear_algebra::gemm<false, false>(1.0, Fa_, X_, 0.0, Temp1a);
      einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1a, 0.0, &Fta_);

      *Evecsa = Fta_;

      einsums::linear_algebra::syev(Evecsa, Evalsa);

      einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsa, 0.0, &Ca_);
      timer_off("Form Ca");
    }

// Beta
#pragma omp task depend(in : this->Fb_, this->X_)                              \
    depend(out : *Temp1b, this -> Ftb_, this->Cb_, *Evalsb, *Evecsb)
    {
      timer_on("Form Cb");
      einsums::linear_algebra::gemm<false, false>(1.0, Fb_, X_, 0.0, Temp1b);
      einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1b, 0.0, &Ftb_);

      *Evecsb = Ftb_;

      einsums::linear_algebra::syev(Evecsb, Evalsb);

      einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsb, 0.0, &Cb_);
      timer_off("Form Cb");
    }
#pragma omp taskwait depend(in : *Evalsa, *Evalsb, this->Ca_, this->Cb_)       \
    depend(out : this -> Cocca_, this->Coccb_)

    // Update Cocc.
    update_Cocc(*Evalsa, *Evalsb);

    if (aocc_per_irrep_ != old_occsa || bocc_per_irrep_ != old_occsb) {
      outfile->Printf("    Occupation Changed:\n         \t");

      for (int i = 0; i < S_.num_blocks(); i++) {
        outfile->Printf("%4s\t", S_.name(i).c_str());
      }

      outfile->Printf("\n    AOCC \t");
      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", aocc_per_irrep_[i]);
      }

      outfile->Printf("\n    BOCC \t");
      for (int i = 0; i < bocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", bocc_per_irrep_[i]);
      }

      outfile->Printf("\n    VIRT \t");

      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i] - aocc_per_irrep_[i]);
      }

      outfile->Printf("\n    Total\t");

      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i]);
      }

      outfile->Printf("\n");
    }

    old_occsa = aocc_per_irrep_;
    old_occsb = bocc_per_irrep_;

#pragma omp task depend(in : this->Cocca_) depend(out : this -> Da_)
    {
      timer_on("Form Da");
      einsums::tensor_algebra::einsum(
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          &Da_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::m},
          Cocca_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                           einsums::tensor_algebra::index::m},
          Cocca_);
      timer_off("Form Da");
    }

#pragma omp task depend(in : this->Coccb_) depend(out : this -> Db_)
    {
      timer_on("Form Db");
      einsums::tensor_algebra::einsum(
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          &Db_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::m},
          Coccb_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                           einsums::tensor_algebra::index::m},
          Coccb_);
      timer_off("Form Db");
    }

    // Optional printing
    if (print_ > 3) {
#pragma omp taskwait
      fprintln(*outfile->stream(), Fta_);
      fprintln(*outfile->stream(), Fa_);
      fprintln(*outfile->stream(), *Evecsa);
      fprintln(*outfile->stream(), *Evalsa);
      fprintln(*outfile->stream(), Ca_);
      fprintln(*outfile->stream(), Da_);
      fprintln(*outfile->stream(), *FDSa);
      fprintln(*outfile->stream(), *SDFa);
      Temp1a->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1a);

      fprintln(*outfile->stream(), Ftb_);
      fprintln(*outfile->stream(), Fb_);
      fprintln(*outfile->stream(), *Evecsb);
      fprintln(*outfile->stream(), *Evalsb);
      fprintln(*outfile->stream(), Cb_);
      fprintln(*outfile->stream(), Db_);
      fprintln(*outfile->stream(), *FDSb);
      fprintln(*outfile->stream(), *SDFb);
      Temp1b->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1b);
    }

    iter++;
  }
  outfile->Printf(
      "    *===========================================================*\n");

  if (!converged)
    throw PSIEXCEPTION("The SCF iterations did not converge.");

  // Compute S^2.
  einsums::DeviceTensor<double, 0> spin;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &spin, 0.5,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_);

  einsums::tensor_algebra::einsum(
      1.0, einsums::tensor_algebra::Indices{}, &spin, -0.5,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_);

  double s_squared = (double)spin * ((double)spin + 1);
  double s2_expected =
      (molecule_->multiplicity() + 1) * (molecule_->multiplicity() - 1) / 4.0;

  outfile->Printf("Spin contamination: %lf\nS^2 value: %lf\nS^2 expected: "
                  "%lf\nS expected: %lf\nS observed: %lf\n",
                  s_squared - s2_expected, s_squared, s2_expected,
                  (molecule_->multiplicity() - 1) / 2.0, (double)spin);

  Evalsa->set_name("Alpha Orbital Energies");
  outfile->Printf("\nAlpha Occupied:\n");

  std::vector<int> inds(nirrep_);
  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < naocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= aocc_per_irrep_[j]) {
        continue;
      }
      if ((*Evalsa)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evalsa)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Alpha Unoccupied:\n");

  for (int i = naocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if ((*Evalsa)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evalsa)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }
  Evalsb->set_name("Beta Orbital Energies");

  outfile->Printf("\n\nBeta Occupied:\n");

  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < nbocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= bocc_per_irrep_[j]) {
        continue;
      }
      if ((*Evalsb)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evalsb)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Beta Unoccupied:\n");

  for (int i = nbocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if ((*Evalsb)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evalsb)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  energy_ = e_new;

  delete coefsa;
  delete focksa;
  delete errorsa;

  delete Temp1a;
  delete Temp2a;
  delete Evecsa;
  delete Evalsa;
  delete FDSa;
  delete SDFa;
  delete Ja;
  delete Ka;

  delete coefsb;
  delete focksb;
  delete errorsb;

  delete Temp1b;
  delete Temp2b;
  delete Evecsb;
  delete Evalsb;
  delete FDSb;
  delete SDFb;
  delete Jb;
  delete Kb;

  delete dRMS_tensa;
  delete dRMS_tensb;

  delete error_valsa;
  delete error_valsb;
  delete elec_a;
  delete elec_b;

  return e_new;
}
void GPUEinsumsUHF::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums SCF\n");
  outfile->Printf("                                by Connor Briggs\n");
  outfile->Printf("                                 %4s Reference\n",
                  options_.get_str("REFERENCE").c_str());
  outfile->Printf("                               Running on the CPU\n");
  outfile->Printf("                      %3d Threads, %6ld MiB Core\n", nthread,
                  memory_ / 1048576L);
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("\n");
  outfile->Printf("  ==> Geometry <==\n\n");

  molecule_->print();

  //    outfile->Printf("  Running in %s symmetry.\n\n",
  //    molecule_->point_group()->symbol().c_str());

  molecule_->print_rotational_constants();

  outfile->Printf("  Nuclear repulsion = %20.15f\n\n",
                  molecule_->nuclear_repulsion_energy({0, 0, 0}));
  // outfile->Printf("  Charge       = %d\n", charge_);
  // outfile->Printf("  Multiplicity = %d\n", multiplicity_);
  outfile->Printf("  Nalpha       = %d\n", nalpha_);
  outfile->Printf("  Nbeta        = %d\n\n", nbeta_);

  outfile->Printf("  ==> Algorithm <==\n\n");
  outfile->Printf("  SCF Algorithm Type is %s.\n",
                  options_.get_str("SCF_TYPE").c_str());
  outfile->Printf("  DIIS %s.\n",
                  options_.get_bool("DIIS") ? "enabled" : "disabled");
  outfile->Printf("  Energy threshold   = %3.2e\n",
                  options_.get_double("E_CONVERGENCE"));
  outfile->Printf("  Density threshold  = %3.2e\n",
                  options_.get_double("D_CONVERGENCE"));

  outfile->Printf("  ==> Primary Basis <==\n\n");

  basisset_->print_by_level("outfile", print_);
}
} // namespace einhf
} // namespace psi
