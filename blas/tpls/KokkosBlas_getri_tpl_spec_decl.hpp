//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSBLAS_GETRI_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GETRI_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {
template <class AViewType, class PViewType, class WorkViewType>
inline void getri_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  printf("KokkosBlas::getri<> TPL MAGMA specialization for < %s , %s, %s >\n",
         typeid(AViewType).name(), typeid(PViewType).name(), typeid(WorkViewType).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  printf("KokkosBlas::getri<> TPL Blas specialization for < %s , %s, %s >\n",
         typeid(AViewType).name(), typeid(PViewType).name(), typeid(WorkViewType).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include <KokkosBlas_Host_tpl.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                               \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_BLAS,double]");    \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      const int N     = static_cast<int>(A.extent(1));                        \
      const int AST   = static_cast<int>(A.stride(1));                        \
      const int LDA   = (AST == 0) ? 1 : AST;                                 \
      const int lwork = static_cast<int>(work.extent(0));                     \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<double>::getri(N, A.data(), LDA, IPIV.data(), work.data(),     \
                              lwork, info);                                   \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                               \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_BLAS,float]");     \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      const int N     = static_cast<int>(A.extent(1));                        \
      const int AST   = static_cast<int>(A.stride(1));                        \
      const int LDA   = (AST == 0) ? 1 : AST;                                 \
      const int lwork = static_cast<int>(work.extent(0));                     \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<float>::getri(N, A.data(), LDA, IPIV.data(), work.data(),      \
                             lwork, info);                                    \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<int*, LAYOUT,                                     \
                            Kokkos::Device<Kokkos::DefaultHostExecutionSpace, \
                                           Kokkos::HostSpace>,                \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUT,                 \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_BLAS,complex<double>]");                     \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      const int N     = static_cast<int>(A.extent(1));                        \
      const int AST   = static_cast<int>(A.stride(1));                        \
      const int LDA   = (AST == 0) ? 1 : AST;                                 \
      const int lwork = static_cast<int>(work.extent(0));                     \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<std::complex<double> >::getri(                                 \
          N, reinterpret_cast<std::complex<double>*>(A.data()), LDA,          \
          IPIV.data(), reinterpret_cast<std::complex<double>*>(work.data()),  \
          lwork, info);                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                 \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<int*, LAYOUT,                                     \
                            Kokkos::Device<Kokkos::DefaultHostExecutionSpace, \
                                           Kokkos::HostSpace>,                \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>*, LAYOUT,                  \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_BLAS,complex<float>]");                      \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      const int N     = static_cast<int>(A.extent(1));                        \
      const int AST   = static_cast<int>(A.stride(1));                        \
      const int LDA   = (AST == 0) ? 1 : AST;                                 \
      const int lwork = static_cast<int>(work.extent(0));                     \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      HostBlas<std::complex<float> >::getri(                                  \
          N, reinterpret_cast<std::complex<float>*>(A.data()), LDA,           \
          IPIV.data(), reinterpret_cast<std::complex<float>*>(work.data()),   \
          lwork, info);                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_SGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_ZGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_CGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                               \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_MAGMA,double]");   \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      magma_int_t N     = static_cast<magma_int_t>(A.extent(1));              \
      magma_int_t AST   = static_cast<magma_int_t>(A.stride(1));              \
      magma_int_t LDA   = (AST == 0) ? 1 : AST;                               \
      magma_int_t lwork = static_cast<int>(work.extent(0));                   \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_dgetri_gpu(N, reinterpret_cast<magmaDouble_ptr>(A.data()), LDA,   \
                       reinterpret_cast<magmaInt_ptr>(IPIV.data()),           \
                       reinterpret_cast<magmaDouble_ptr>(work.data()), lwork, \
                       &info);                                                \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                               \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float*, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_MAGMA,float]");    \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      magma_int_t N     = static_cast<magma_int_t>(A.extent(1));              \
      magma_int_t AST   = static_cast<magma_int_t>(A.stride(1));              \
      magma_int_t LDA   = (AST == 0) ? 1 : AST;                               \
      magma_int_t lwork = static_cast<int>(work.extent(0));                   \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_sgetri_gpu(N, reinterpret_cast<magmaFloat_ptr>(A.data()), LDA,    \
                       reinterpret_cast<magmaInt_ptr>(IPIV.data()),           \
                       reinterpret_cast<magmaFloat_ptr>(work.data()), lwork,  \
                       &info);                                                \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<magma_int_t*, LAYOUT,                             \
                            Kokkos::Device<Kokkos::DefaultHostExecutionSpace, \
                                           Kokkos::HostSpace>,                \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUT,                 \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_MAGMA,complex<double>]");                    \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      magma_int_t N     = static_cast<magma_int_t>(A.extent(1));              \
      magma_int_t AST   = static_cast<magma_int_t>(A.stride(1));              \
      magma_int_t LDA   = (AST == 0) ? 1 : AST;                               \
      magma_int_t lwork = static_cast<int>(work.extent(0));                   \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_zgetri_gpu(                                                       \
          N, reinterpret_cast<magmaDoubleComplex_ptr>(A.data()), LDA,         \
          reinterpret_cast<magmaInt_ptr>(IPIV.data()),                        \
          reinterpret_cast<magmaDoubleComplex_ptr>(work.data()),              \
          lwork, &info);                                                      \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                 \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<magma_int_t*, LAYOUT,                             \
                            Kokkos::Device<Kokkos::DefaultHostExecutionSpace, \
                                           Kokkos::HostSpace>,                \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>*, LAYOUT,                  \
                            Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUT,                                     \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WorkViewType;                                                         \
                                                                              \
    static void getri(const AViewType& A, const PViewType& IPIV, const WorkViewType& work) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_MAGMA,complex<float>]");                     \
      getri_print_specialization<AViewType, PViewType, WorkViewType>();       \
                                                                              \
      magma_int_t N     = static_cast<magma_int_t>(A.extent(1));              \
      magma_int_t AST   = static_cast<magma_int_t>(A.stride(1));              \
      magma_int_t LDA   = (AST == 0) ? 1 : AST;                               \
      magma_int_t lwork = static_cast<int>(work.extent(0));                   \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      magma_cgetri_gpu(                                                       \
          N, reinterpret_cast<magmaFloatComplex_ptr>(A.data()), LDA,          \
          reinterpret_cast<magmaInt_ptr>(IPIV.data()),                        \
          reinterpret_cast<magmaFloatComplex_ptr>(work.data()),               \
          lwork, &info);                                                      \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_DGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_SGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_SGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_ZGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_ZGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_CGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_CGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif
