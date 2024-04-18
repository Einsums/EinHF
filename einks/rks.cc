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

#include "rks.h"

#include "einsums.hpp"

#include <optional>
#include <map>
#include <string>

#include "psi4/libfock/jk.h"
#include "psi4/libfock/v.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/libfunctional/superfunctional.h"
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
namespace einks {

EinsumsRKS::EinsumsRKS(SharedWavefunction ref_wfn, Options &options)
    : Wavefunction(options) {

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();

  print_ = options_.get_int("PRINT");
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
  F_.set_name("Fock Matrix");
  Ft_.set_name("Transformed Fock Matrix");
  C_.set_name("MO Coefficients");
  Cocc_.set_name("Occupied MO Coefficients");
  D_.set_name("Density Matrix");

  for (int i = 0; i < this->nirrep_; i++) {
    X_.push_block(einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                             irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    F_.push_block(einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                             irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    this->Ft_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    C_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    Cocc_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    D_.push_block(block);
  }
}

EinsumsRKS::~EinsumsRKS() {}

void EinsumsRKS::init_integrals() {
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
  if (nelec % 2) {
    throw PSIEXCEPTION("This is only an RHF code, but you gave it an odd "
                       "number of electrons.  Try again!");
  }
  ndocc_ = nelec / 2;

  outfile->Printf("    There are %d doubly occupied orbitals\n", ndocc_);
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

  occ_per_irrep_.resize(nirrep_);

  H_mat->add(mints->so_potential());

  H_.set_name("Hamiltonian");
  S_.set_name("Overlap");

  for (int i = 0; i < S_mat->nirrep(); i++) {
    irrep_sizes_.push_back(S_mat->rowdim(i));
    auto S_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), S_mat->rowdim(i), S_mat->coldim(i));

    for (int j = 0; j < S_mat->rowdim(i); j++) {
      for (int k = 0; k < S_mat->coldim(i); k++) {
        S_block(j, k) = S_mat->get(i, j, k);
      }
    }

    S_.push_block(S_block);
  }

  for (int i = 0; i < S_mat->nirrep(); i++) {
    auto H_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), H_mat->rowdim(i), H_mat->coldim(i));

    for (int j = 0; j < H_mat->rowdim(i); j++) {
      for (int k = 0; k < H_mat->coldim(i); k++) {
        H_block(j, k) = H_mat->get(i, j, k);
      }
    }

    H_.push_block(H_block);
  }

  if (print_ > 3) {
    fprintln(*outfile->stream(), S_);
    fprintln(*outfile->stream(), H_);
    outfile->stream()->flush();
  }

  outfile->Printf("    Forming JK object\n\n");
  // Construct a JK object that compute J and K SCF matrices very efficiently
  jk_ = JK::build_JK(basisset_, mintshelper_->get_basisset("DF_BASIS_SCF"),
                     options_, false, Process::environment.get_memory() * 0.8);
  jk_->set_do_K(false);

  // This is a very heavy compute object, lets give it 80% of our total memory
  // jk_->set_memory(Process::environment.get_memory() * 0.8);
  jk_->initialize();
  jk_->print_header();

  std::optional<std::map<std::string, double>> tweaks(std::in_place);
  func_ = std::make_shared<SuperFunctional>();
  //func_ = SuperFunctional::XC_build(options_.get_str("DFT_FUNCTIONAL"), true, tweaks);

  v_ = VBase::build_V(basisset_, func_, options_);
  v_->initialize();
  v_->print_header();
}

double EinsumsRKS::compute_electronic_energy() {
  // Compute the electronic energy: (H + F)_pq * D_pq -> energy

  einsums::Tensor<double, 0> e_tens;
  auto temp = einsums::BlockTensor<double, 2>("temp", H_.vector_dims());

  temp = H_;
  temp += F_;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &e_tens, 1.0,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      temp);

  return (double)e_tens;
}

void EinsumsRKS::update_Cocc(const einsums::Tensor<double, 1> &energies) {
  // Update occupation.

  for (int i = 0; i < nirrep_; i++) {
    occ_per_irrep_[i] = 0;
  }

  for (int i = 0; i < ndocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (occ_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = energies(S_.block_range(j)[0] + occ_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    occ_per_irrep_[irrep_occ]++;
  }

  Cocc_.zero();

#pragma omp taskgroup
  {
#pragma omp taskloop
    for (int i = 0; i < nirrep_; i++) {
      Cocc_[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep_[i])) =
          C_[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep_[i]));
    }
  }
}

void EinsumsRKS::compute_diis_coefs(
    const std::deque<einsums::BlockTensor<double, 2>> &errors,
    std::vector<double> *out) const {
  einsums::Tensor<double, 2> B_mat = einsums::Tensor<double, 2>(
      "DIIS error matrix", errors.size() + 1, errors.size() + 1);

  B_mat.zero();
  B_mat(einsums::Range{errors.size(), errors.size() + 1},
        einsums::Range{0, errors.size()}) = 1.0;
  B_mat(einsums::Range{0, errors.size()},
        einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
  for (int i = 0; i < errors.size(); i++) {
    for (int j = 0; j <= i; j++) {
      B_mat(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
      B_mat(j, i) = B_mat(i, j);
    }
  }

  einsums::Tensor<double, 2> res_mat =
      einsums::Tensor<double, 2>("DIIS result matrix", 1, errors.size() + 1);

  res_mat.zero();
  res_mat(0, errors.size()) = 1.0;

  einsums::linear_algebra::gesv(&B_mat, &res_mat);

  out->resize(errors.size());

  for (int i = 0; i < errors.size(); i++) {
    out->at(i) = res_mat(0, i);
  }
}

void EinsumsRKS::compute_diis_fock(
    const std::vector<double> &coefs,
    const std::deque<einsums::BlockTensor<double, 2>> &focks,
    einsums::BlockTensor<double, 2> *out) const {

  out->zero();

#pragma omp parallel for
  for (int i = 0; i < coefs.size(); i++) {
    einsums::linear_algebra::axpy(coefs[i], focks[i], out);
  }
}

double EinsumsRKS::compute_energy() {
  if (diis_max_iters_ != 0) {
    outfile->Printf("Performing DIIS with %d vectors.\n", diis_max_iters_);
  } else {
    outfile->Printf("Turning DIIS off.\n");
  }

  // Allocate a few temporary matrices
  auto Temp1 =
      new einsums::BlockTensor<double, 2>("Temporary Array 1", irrep_sizes_);
  auto Temp2 =
      new einsums::BlockTensor<double, 2>("Temporary Array 2", irrep_sizes_);
  auto FDS = new einsums::BlockTensor<double, 2>("FDS", irrep_sizes_);
  auto SDF = new einsums::BlockTensor<double, 2>("SDF", irrep_sizes_);
  auto Evecs =
      new einsums::BlockTensor<double, 2>("Eigenvectors", irrep_sizes_);
  auto Evals = new einsums::Tensor<double, 1>("Eigenvalues", nso_);

  std::deque<einsums::BlockTensor<double, 2>>
      *errors = new std::deque<einsums::BlockTensor<double, 2>>(0),
      *focks = new std::deque<einsums::BlockTensor<double, 2>>(0);
  std::vector<double> *coefs = new std::vector<double>(0);

  // Form the X_ matrix (S^-1/2)
  X_ = einsums::linear_algebra::pow(S_, -0.5);

  F_ = H_;

  einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, Temp1);
  einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1, 0.0, &Ft_);

  *Evecs = Ft_;

  Evals->zero();
  einsums::linear_algebra::syev(Evecs, Evals);

  einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecs, 0.0, &C_);

  update_Cocc(*Evals);

  einsums::tensor_algebra::einsum(
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      &D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::m},
      Cocc_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                       einsums::tensor_algebra::index::m},
      Cocc_);

  if (print_ > 3) {
    outfile->Printf(
        "MO Coefficients and density from Core Hamiltonian guess:\n");
    fprintln(*outfile->stream(), X_);
    fprintln(*outfile->stream(), C_);
    fprintln(*outfile->stream(), D_);
    fprintln(*outfile->stream(), *Evals);
    fprintln(*outfile->stream(), Cocc_);
  }

  int iter = 1;
  bool converged = false;
  double e_old;
  double e_new = e_nuc_ + compute_electronic_energy();

  outfile->Printf("    Energy from core Hamiltonian guess: %20.16f\n\n", e_new);

  outfile->Printf(
      "    *===========================================================*\n");
  outfile->Printf(
      "    * Iter       Energy            delta E    ||gradient||      *\n");
  outfile->Printf(
      "    *-----------------------------------------------------------*\n");

  while (!converged && iter < maxiter_) {
    e_old = e_new;

    // Add the core Hamiltonian term to the Fock operator
    F_ = H_;

    // The JK object handles all of the two electron integrals
    // To enhance efficiency it does use the density, but the orbitals
    // themselves D_uv = C_ui C_vj J_uv = I_uvrs D_rs K_uv = I_urvs D_rs

    // Here we clear the old Cocc and push_back our new one
    std::vector<SharedMatrix> &Cl = jk_->C_left();
    Cl.clear();

    SharedMatrix CTemp = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                  occ_per_irrep_.data());

    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < occ_per_irrep_[i]; k++) {
          (*CTemp.get())(i, j, k) = Cocc_[i](j, k);
        }
      }
    }

    Cl.push_back(CTemp);
    jk_->compute();

    // Obtain the new J and K matrices
    const std::vector<SharedMatrix> &J_mat = jk_->J();

    // Proceede as normal
    auto J = einsums::BlockTensor<double, 2>("J matrix", irrep_sizes_);

    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < irrep_sizes_[i]; k++) {
          J[i](j, k) = J_mat[0]->get(i, j, k);
        }
      }
    }

    F_ += J;

    SharedMatrix D_mat = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                  occ_per_irrep_.data());
    
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < occ_per_irrep_[i]; k++) {
          (*D_mat.get())(i, j, k) = D_[i](j, k);
        }
      }
    }

    auto V = einsums::BlockTensor<double, 2>("V matrix", irrep_sizes_);

    std::vector<SharedMatrix> D_vec{D_mat};
    std::vector<SharedMatrix> V_vec(1);

    v_->set_D(D_vec);
    v_->compute_V(V_vec);

    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < irrep_sizes_[i]; k++) {
          V[i](j, k) = V_vec[0]->get(i, j, k);
        }
      }
    }

    F_ += V;

    // Compute the orbital gradient, FDS-SDF
    einsums::linear_algebra::gemm<false, false>(1.0, D_, S_, 0.0, Temp1);
    einsums::linear_algebra::gemm<false, false>(1.0, F_, *Temp1, 0.0, FDS);
    einsums::linear_algebra::gemm<false, false>(1.0, D_, F_, 0.0, Temp1);
    einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp1, 0.0, SDF);

    *Temp1 = *FDS;
    *Temp1 -= *SDF;

    // Density RMS
    einsums::Tensor<double, 0> *dRMS_tens = new einsums::Tensor<double, 0>();
    *dRMS_tens = 0;

#pragma omp taskgroup
    {
#pragma omp task depend(in : *Temp1)                                           \
    depend(inout : *errors, *focks, this->F_, coefs)
      {
        if (diis_max_iters_ > 0) {

          if (errors->size() == diis_max_iters_) {
            errors->pop_front();
          }
          errors->push_back(*Temp1);
          if (focks->size() == diis_max_iters_) {
            focks->pop_front();
          }
          focks->push_back(F_);

          compute_diis_coefs(*errors, coefs);

          compute_diis_fock(*coefs, *focks, &F_);
        }
      }

#pragma omp task depend(in : *Temp1) depend(out : *dRMS_tens)
      {
        einsums::tensor_algebra::einsum(
            0.0, einsums::tensor_algebra::Indices{}, dRMS_tens,
            1.0 / (nso_ * nso_),
            einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                             einsums::tensor_algebra::index::j},
            *Temp1,
            einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                             einsums::tensor_algebra::index::j},
            *Temp1);
      }
    }

    double dRMS = std::sqrt((double)*dRMS_tens);

    // Compute the energy
    e_new = e_nuc_ + compute_electronic_energy();
    double dE = e_new - e_old;

    // Optional printing
    if (print_ > 3) {
      fprintln(*outfile->stream(), Ft_);
      fprintln(*outfile->stream(), F_);
      fprintln(*outfile->stream(), *Evecs);
      fprintln(*outfile->stream(), *Evals);
      fprintln(*outfile->stream(), C_);
      fprintln(*outfile->stream(), D_);
      fprintln(*outfile->stream(), *FDS);
      fprintln(*outfile->stream(), *SDF);
      Temp1->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1);

      outfile->Printf("DIIS error size: %d\nDIIS Focks size: %d\n",
                      errors->size(), focks->size());
      outfile->Printf("DIIS coefs: ");

      for (int i = 0; i < coefs->size(); i++) {
        outfile->Printf("%lf ", coefs->at(i));
      }
      outfile->Printf("\n");
    }

    converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

    outfile->Printf("    * %3d %20.14f    %9.2e    %9.2e    ", iter, e_new, dE,
                    dRMS);
    if (focks->size() > 0) {
      outfile->Printf("DIIS*\n");
    } else {
      outfile->Printf("    *\n");
    }

    einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, Temp1);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1, 0.0, &Ft_);

    *Evecs = Ft_;
    einsums::linear_algebra::syev(Evecs, Evals);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecs, 0.0, &C_);

    update_Cocc(*Evals);

    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &D_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Cocc_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Cocc_);
    iter++;
  }
  outfile->Printf(
      "    *===========================================================*\n");

  if (!converged)
    throw PSIEXCEPTION("The SCF iterations did not converge.");

  Evals->set_name("Orbital Energies");
  outfile->Printf("\nOccupied:\n");

  std::vector<int> inds(nirrep_);
  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < ndocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= occ_per_irrep_[j]) {
        continue;
      }
      if ((*Evals)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evals)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Unoccupied:\n");

  for (int i = ndocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if ((*Evals)(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = (*Evals)(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }
  energy_ = e_new;

  delete coefs;
  delete focks;
  delete errors;

  delete Temp1;
  delete Temp2;
  delete Evecs;
  delete Evals;
  delete FDS;
  delete SDF;

  return e_new;
}
} // namespace einks
} // namespace psi
