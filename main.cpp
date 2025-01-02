#include <iostream>
#include <ranges>
#include <array>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <autodiff/forward/dual/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>


template <typename ST, int size, int Options, int maxSize>
requires((size == 1 or size == 3 or size == 6) or
         ((maxSize == 1 or maxSize == 3 or maxSize == 6) and size == Eigen::Dynamic))
auto fromVoigt(const Eigen::Matrix<ST, size, 1, Options, maxSize, 1>& EVoigt, bool isStrain = true) {
  constexpr bool isFixedSized   = (size != Eigen::Dynamic);
  const ST possibleStrainFactor = isStrain ? 0.5 : 1.0;

  const size_t inputSize = isFixedSized ? size : EVoigt.size();
  const size_t matrixSize =      3;

  auto E = [&]() {
    if constexpr (isFixedSized) {
      Eigen::Matrix<ST, matrixSize, matrixSize> E;
      E.diagonal() = EVoigt.template head<matrixSize>();
      return E;
    } else {
      Eigen::Matrix<ST, Eigen::Dynamic, Eigen::Dynamic, Options, 3, 3> E;
      E.resize(matrixSize, matrixSize);
      E.diagonal() = EVoigt.template head(matrixSize);
      return E;
    }
  }();

  if (matrixSize == 2) {
    E(0, 1) = E(1, 0) = EVoigt(2) * possibleStrainFactor;
  } else if (matrixSize == 3) {
    E(2, 1) = E(1, 2) = EVoigt(matrixSize) * possibleStrainFactor;
    E(2, 0) = E(0, 2) = EVoigt(matrixSize + 1) * possibleStrainFactor;
    E(1, 0) = E(0, 1) = EVoigt(matrixSize + 2) * possibleStrainFactor;
  }

  return E;
}

template <typename ST, int size, int Options, int maxSize>
requires((size > 0 and size <= 3) or (maxSize > 0 and maxSize <= 3 and size == Eigen::Dynamic))
auto toVoigt(const Eigen::Matrix<ST, size, size, Options, maxSize, maxSize>& E, bool isStrain = true) {
  constexpr bool isFixedSized   = (size != Eigen::Dynamic);
  const ST possibleStrainFactor = isStrain ? 2.0 : 1.0;

  const size_t inputSize = isFixedSized ? size : E.rows();
  auto EVoigt            = [&]() {
    if constexpr (isFixedSized) {
      Eigen::Vector<ST, (size * (size + 1)) / 2> EVoigt;
      EVoigt.template head<size>() = E.diagonal();
      return EVoigt;
    } else {
      Eigen::Matrix<ST, Eigen::Dynamic, 1, Options, 6, 1> EVoigt;
      EVoigt.resize((inputSize * (inputSize + 1)) / 2);
      EVoigt.template head(inputSize) = E.diagonal();
      return EVoigt;
    }
  }();

  if (inputSize == 2)
    EVoigt(2) = E(0, 1) * possibleStrainFactor;
  else if (inputSize == 3) {
    EVoigt(inputSize)     = E(1, 2) * possibleStrainFactor;
    EVoigt(inputSize + 1) = E(0, 2) * possibleStrainFactor;
    EVoigt(inputSize + 2) = E(0, 1) * possibleStrainFactor;
  }
  return EVoigt;
}

template <typename ST, int n>
struct OgdenT
{
  using ScalarType         = ST;
  using PrincipalStretches = Eigen::Vector<ScalarType, 3>;

  static constexpr int numMatParameters           = n;
  static constexpr int dim                        = 3;

  using FirstDerivative  = Eigen::Vector<ScalarType, dim>;
  using SecondDerivative = Eigen::Matrix<ScalarType, dim, dim>;

  using MaterialParameters = std::array<double, numMatParameters>;
  using MaterialExponents  = std::array<double, numMatParameters>;

  /**
   * \brief Constructor for OgdenT
   *
   * \param mpt material parameters (array of mu values)
   * \param mex material exponents (array of alpha values)
   */
  explicit OgdenT(const MaterialParameters& mpt, const MaterialExponents& mex)
      : materialParameters_{mpt},
        materialExponents_{mex} {}

  /**
   * \brief Returns the material parameters (mu values) stored in the material
   */
  MaterialParameters materialParametersImpl() const { return materialParameters_; }

  /**
   * \brief Returns the material exponents (alpha values) stored in the material
   */
  const MaterialExponents& materialExponents() const { return materialExponents_; }

  /**
   * \brief Computes the stored energy in the Ogden material model.
   *
   * \param lambda principal stretches
   * \return ScalarType
   */
  ScalarType storedEnergyImpl(const PrincipalStretches& lambda) const {
    auto& mu    = materialParameters_;
    auto& alpha = materialExponents_;

    ScalarType energy{};

    // if constexpr (usesDeviatoricStretches) {
    //   auto lambdaBar = Impl::deviatoricStretches(lambda);

    //   for (auto i : parameterRange())
    //     energy += mu[i] / alpha[i] *
    //               (pow(lambdaBar[0], alpha[i]) + pow(lambdaBar[1], alpha[i]) + pow(lambdaBar[2], alpha[i]) - 3);

    // } else {
      auto J = lambda[0] * lambda[1] * lambda[2];

      for (auto i : parameterRange()) {
        energy +=
            mu[i] / alpha[i] * (pow(lambda[0], alpha[i]) + pow(lambda[1], alpha[i]) + pow(lambda[2], alpha[i]) - 3) -
            mu[i] * log(J);
      }
    // }
    return energy;
  }

  /**
   * \brief Computes the first derivative of the stored energy function w.r.t. the total principal stretches.
   *
   * \param lambda principal stretches
   * \return ScalarType
   */
  FirstDerivative firstDerivativeImpl(const PrincipalStretches& lambda) const {
    auto& mu       = materialParameters_;
    auto& alpha    = materialExponents_;
    auto dWdLambda = FirstDerivative::Zero().eval();

    // if constexpr (usesDeviatoricStretches) {
    //   auto lambdaBar = Impl::deviatoricStretches(lambda);

    //   auto dWdLambdaBar = FirstDerivative::Zero().eval();
    //   for (const auto j : parameterRange())
    //     for (const auto k : dimensionRange())
    //       dWdLambdaBar[k] += mu[j] * (pow(lambdaBar[k], alpha[j] - 1));

    //   ScalarType sumLambdaBar{0.0};
    //   for (const auto b : dimensionRange())
    //     sumLambdaBar += lambdaBar[b] * dWdLambdaBar[b];

    //   for (const auto i : dimensionRange())
    //     dWdLambda[i] = (lambdaBar[i] * dWdLambdaBar[i] - (1.0 / 3.0) * sumLambdaBar) / lambda[i];

    // } else {
      for (const auto j : parameterRange())
        for (const auto k : dimensionRange())
          dWdLambda[k] += (mu[j] * (pow(lambda[k], alpha[j]) - 1)) / lambda[k];
    // }
    return dWdLambda;
  }

  /**
   * \brief Computes the second derivatives of the stored energy function w.r.t. the total principal stretches.
   *
   * \param lambda principal stretches
   * \return ScalarType
   */
  SecondDerivative secondDerivativeImpl(const PrincipalStretches& lambda) const {
    auto& mu    = materialParameters_;
    auto& alpha = materialExponents_;
    auto dS     = SecondDerivative::Zero().eval();

    // if constexpr (usesDeviatoricStretches) {
    //   const auto lambdaBar = Impl::deviatoricStretches(lambda);
    //   const auto dWdLambda = firstDerivativeImpl(lambda);

    //   for (const auto a : dimensionRange()) {
    //     for (const auto b : dimensionRange()) {
    //       if (a == b) {
    //         for (const auto p : parameterRange()) {
    //           ScalarType sumC{0.0};
    //           for (auto c : dimensionRange())
    //             sumC += pow(lambdaBar[c], alpha[p]);
    //           dS(a, b) += mu[p] * alpha[p] * ((1.0 / 3.0) * pow(lambdaBar[a], alpha[p]) + (1.0 / 9.0) * sumC);
    //         }
    //       } else {
    //         for (const auto p : parameterRange()) {
    //           ScalarType sumC{0.0};
    //           for (auto c : dimensionRange())
    //             sumC += pow(lambdaBar[c], alpha[p]);
    //           dS(a, b) +=
    //               mu[p] * alpha[p] *
    //               (-(1.0 / 3.0) * (pow(lambdaBar[a], alpha[p]) + pow(lambdaBar[b], alpha[p])) + (1.0 / 9.0) * sumC);
    //         }
    //       }

    //       dS(a, b) *= 1.0 / (lambda[a] * lambda[b]);

    //       if (a == b)
    //         dS(a, b) -= (2.0 / lambda[a]) * dWdLambda[a];
    //     }
    //   }
    // } else {
      for (const auto j : parameterRange())
        for (const auto k : dimensionRange())
          dS(k, k) += (-2 * (mu[j] * (pow(lambda[k], alpha[j]) - 1)) + (mu[j] * pow(lambda[k], alpha[j]) * alpha[j])) /
                      pow(lambda[k], 2);
    // }
    return dS;
  }

  /**
   * \brief Rebinds the material to a different scalar type.
   * \tparam STO The target scalar type.
   * \return OgdenT<ScalarTypeOther> The rebound Ogden material.
   */
  // template <typename STO>
  // auto rebind() const {
  //   return OgdenT<STO, numMatParameters, stretchTag>(materialParameters_, materialExponents_);
  // }

private:
  MaterialParameters materialParameters_;
  MaterialExponents materialExponents_;

  inline static constexpr auto parameterRange() { return std::ranges::views::iota(0,numMatParameters); }
  inline static constexpr auto dimensionRange() { return std::ranges::views::iota(0,dim); }
};


 namespace Impl {
    template <typename T>
    struct is_dual : std::false_type
    {
    };

    // Specialization for Dual<T, U>: this will be true for Dual types
    template <typename T, typename U>
    struct is_dual<autodiff::detail::Dual<T, U>> : std::true_type
    {
    };
  } // namespace Impl

  namespace Concepts{
 template <typename T>
  concept AutodiffScalar = Impl::is_dual<T>::value;
  }
template < typename Derived>
auto principalStretches(const Eigen::MatrixBase<Derived>& Cvec, int options = Eigen::ComputeEigenvectors) {
 Eigen::Matrix<typename Derived::Scalar,3,3> C = fromVoigt(Cvec.derived().eval());
  using DerivedC = decltype(C); 
   using ScalarType = DerivedC::Scalar;
  Eigen::SelfAdjointEigenSolver<DerivedC> eigensolver{};
 

  // For AD we use the direct method which uses a closed form calculation, but has less accuracy
eigensolver.compute(C, options);

  if (eigensolver.info() != Eigen::Success)
    std::cerr<< "Failed to compute eigenvalues and eigenvectors of C.";

  auto& eigenvalues  = eigensolver.eigenvalues();
  auto& eigenvectors = options == Eigen::ComputeEigenvectors ? eigensolver.eigenvectors() : DerivedC::Zero();

  auto principalStretches = eigenvalues.array().sqrt().eval();
  return std::make_pair(principalStretches, eigenvectors);
}

template <typename Derived, typename T, auto rank>
Eigen::Tensor<typename Derived::Scalar, rank> tensorView(const Eigen::EigenBase<Derived>& matrix,
                                                         const std::array<T, rank>& dims) {
  return Eigen::TensorMap<const Eigen::TensorFixedSize<
      const typename Derived::Scalar, Eigen::Sizes<Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>>(
      matrix.derived().eval().data(), dims);
}

auto dyadic(const auto& A_ij, const auto& B_kl) {
  Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
  return A_ij.contract(B_kl, empty_index_list).eval();
}
template <typename ST_, int size, bool asTensor = true>
auto dyadic(const Eigen::Vector<ST_, size>& a, const Eigen::Vector<ST_, size>& b) {
  if constexpr (asTensor)
    return tensorView((a * b.transpose()).eval(), std::array<Eigen::Index, 2>({size, size}));
  else
    return (a * b.transpose()).eval();
}

// Unpack the derivatives from the result of an @ref eval call into an array.
template<typename D,typename F>
auto forEach(const Eigen::MatrixBase<D>& result,F&& f)
{
    auto& r = result.derived();
    return r.unaryExpr(f);
}


struct DeviatoricMaterial
{

    std::array<double,1> para{1000};
    std::array<double,1> exp{2.0};

  template <typename Derived>
  auto energy(const Eigen::MatrixBase<Derived>& Cvec)
  {
    using ST =Derived::Scalar;
    OgdenT<ST,1> mat{para,exp};
       using ScalarType = Derived::Scalar;
   
        // if we enter this function with a matrix C with degenerate eigenvalues and a autodiff::dual2nd scalar type the second order derivative has a singularity
    // Therefore, we refactor the code as follows
     if constexpr (std::is_same_v<ScalarType,double>){
    auto lambdas = principalStretches(Cvec).first;
      return mat.storedEnergyImpl(lambdas);
      }
    else if constexpr (std::is_same_v<ScalarType,autodiff::dual>)

  {   
    autodiff::dual2nd e;
    auto realCVec = derivative<0>(Cvec.derived());
    auto dualCVec = derivative<1>(Cvec.derived());
     auto lambdas = principalStretches(realCVec).first;
    OgdenT<double,1> mat{para,exp};
    e.val= mat.storedEnergyImpl(lambdas);
    e.grad= (this->stresses(realCVec).transpose()/2 * fromVoigt(dualCVec)).trace();
     return e;
     }else if constexpr (std::is_same_v<ScalarType,autodiff::dual2nd>)
  {   
    autodiff::dual2nd e;
    Derived d = Cvec.derived();
     std::cout<<"Cvec.derived()"<<Cvec.derived()<<"\n";
    const auto realCVec = derivative<0>(Cvec.derived());
    const auto dualCVec =  fromVoigt(forEach(Cvec.derived(),[](auto& v){return v.grad.val;}).eval());
    const auto dualCVec2 = fromVoigt(forEach(Cvec.derived(),[](auto& v){return v.val.grad;}).eval());
    const auto lambdas = principalStretches(realCVec).first;
    OgdenT<double,1> mat{para,exp};
    e.val= mat.storedEnergyImpl(lambdas);
    e.grad.val = (this->stresses(realCVec).transpose()/2 * dualCVec).trace();
    e.val.grad = e.grad.val ;
    const auto Cmoduli= this->tangentModuli(realCVec);

    Eigen::array<Eigen::IndexPair<Eigen::Index>, 2> double_contraction = { Eigen::IndexPair<Eigen::Index>(2, 0), Eigen::IndexPair<Eigen::Index>(3, 1) };
    Eigen::array<Eigen::IndexPair<Eigen::Index>, 2> double_contraction2 = { Eigen::IndexPair<Eigen::Index>(0, 0), Eigen::IndexPair<Eigen::Index>(1, 1) };
    const auto tCdual = tensorView(dualCVec,std::array<Eigen::Index, 2>({3, 3}));
    const auto tCdualT = tensorView(dualCVec2,std::array<Eigen::Index, 2>({3, 3}));
    const auto prod = Cmoduli.contract( tCdual,double_contraction);
    const Eigen::Tensor<double, 0> res =tCdualT.contract( prod,double_contraction2); 
    e.grad.grad =res(0)/4.0; // extracting value of zero order tensor
    return e;
  }
  } 

    template <typename Derived>
  auto stresses(const Eigen::MatrixBase<Derived>& Cvec)
  {
        using ST =Derived::Scalar;
    OgdenT<ST,1> mat{para,exp};
    auto [lambdas,N] = principalStretches(Cvec);
Eigen::Vector<ST,3> dWdLambda = mat.firstDerivativeImpl(lambdas);
     Eigen::Vector<ST,3> principalStresses = dWdLambda.array() / lambdas.array();

    Eigen::Matrix<ST,3,3> S= N *principalStresses.asDiagonal() * N.transpose();
    // S.setZero();
    // for (int i =0; i<3 ; ++i)
    // S += principalStresses[i]* N.col(i)*N.col(i).transpose();
    return S;
  }

  Eigen::TensorFixedSize<double, Eigen::Sizes<3, 3, 3, 3>> transformDeviatoricTangentModuli(const Eigen::TensorFixedSize<double, Eigen::Sizes<3, 3, 3, 3>>& L,
                                                  const Eigen::Matrix<double, 3, 3>& N) const {
    Eigen::TensorFixedSize<double, Eigen::Sizes<3, 3, 3, 3>> moduli{};
    moduli.setZero();

    for (const auto i : std::ranges::views::iota(0,3))
      for (const auto k : std::ranges::views::iota(0,3)) {
        // First term: L[i, i, k, k] * ((N[i] ⊗ N[i]) ⊗ (N[k] ⊗ N[k]))
        auto NiNi = dyadic(N.col(i).eval(), N.col(i).eval());
        auto NkNk = dyadic(N.col(k).eval(), N.col(k).eval());

        moduli += L(i, i, k, k) * dyadic(NiNi, NkNk);

        // Second term (only if i != k): L[i, k, i, k] * (N[i] ⊗ N[k] ⊗ (N[i] ⊗ N[k] + N[k] ⊗ N[i]))
        if (i != k) {
          auto NiNk = dyadic(N.col(i).eval(), N.col(k).eval());
          auto NkNi = dyadic(N.col(k).eval(), N.col(i).eval());

          moduli += L(i, k, i, k) * dyadic(NiNk, NiNk + NkNi);
        }
      }

    return moduli;
  }

      template <typename Derived>
  auto tangentModuli(const Eigen::MatrixBase<Derived>& Cvec)
  {
        using ST =Derived::Scalar;
    OgdenT<ST,1> mat{para,exp};
     auto [lambdas,N] = principalStretches(Cvec);
Eigen::Vector<ST,3> dWdLambda = mat.firstDerivativeImpl(lambdas);
     Eigen::Vector<ST,3> principalStresses = dWdLambda.array() / lambdas.array();
   
   auto L = Eigen::TensorFixedSize<double, Eigen::Sizes<3, 3, 3, 3>>{};
    L.setZero();
Eigen::Matrix<ST,3,3> ddWddLambda = mat.secondDerivativeImpl(lambdas);

    for (const auto i : std::ranges::views::iota(0,3))
      for (const auto k : std::ranges::views::iota(0,3))
        L(i, i, k, k) = 1.0 / (lambdas(i) * lambdas(k)) * ddWddLambda(i, k);

    for (const auto i : std::ranges::views::iota(0,3))
      for (const auto k : std::ranges::views::iota(0,3))
        if (i != k) {
          if (std::abs(lambdas(i)- lambdas(k))<1e-8)

             L(i, k, i, k) = 0.5 * (L(i, i, i, i) - L(i, i, k, k));
          else
            L(i, k, i, k) += (principalStresses(i) - principalStresses(k)) / (pow(lambdas(i), 2) - pow(lambdas(k), 2));
        }

  const auto moduliDev = transformDeviatoricTangentModuli(L, N);
    return moduliDev;
  }
};


template < typename Derived>
auto principalStretchesAD(const Eigen::MatrixBase<Derived>& Cvec) {
  // auto C= Eigen::Reshaped<const Eigen::Vector<double,9>, 3, 3>(Cvec.derived().eval()).eval();
  auto C0 = fromVoigt(derivative<0>(Cvec.eval()));
  auto C1 = fromVoigt(derivative<1>(Cvec.eval()));
  auto C2 = fromVoigt(derivative<2>(Cvec.eval()));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver{};
  using ScalarType = Derived::Scalar;

  // For AD we use the direct method which uses a closed form calculation, but has less accuracy
    eigensolver.computeDirect(C0);

  if (eigensolver.info() != Eigen::Success)
    std::cerr<< "Failed to compute eigenvalues and eigenvectors of C.";

  auto& eigenvalues  = eigensolver.eigenvalues();
  auto& eigenvectors = eigensolver.eigenvectors() ;

  Eigen::Vector<autodiff::dual2nd,3> principalStretches = eigenvalues.array().sqrt().eval();
  autodiff::dual d;
  d.val=0;
  d.grad=1;
  for(int i = 0; i<3 ; ++i)
    principalStretches[i].grad = 1/(2*sqrt(eigenvalues[i]))*eigenvectors.col(i).dot((C1)*eigenvectors.col(i));

  return std::make_pair(principalStretches, eigenvectors);
}

constexpr Eigen::Index toVoigt(Eigen::Index i, Eigen::Index j) noexcept {
  if (i == j) // _00 -> 0, _11 -> 1,  _22 -> 2
    return i;
  if ((i == 1 and j == 2) or (i == 2 and j == 1)) // _12 and _21 --> 3
    return 3;
  if ((i == 0 and j == 2) or (i == 2 and j == 0)) // _02 and _20 --> 4
    return 4;
  if ((i == 0 and j == 1) or (i == 1 and j == 0)) // _01 and _10 --> 5
    return 5;
  assert(i < 3 and j < 3 && "For Voigt notation the indices need to be 0,1 or 2.");
  __builtin_unreachable();
}

/**
 * \brief Converts a fourth-order tensor of fixed size 3x3x3x3 to a Voigt notation matrix of size 6x6.
 *  \ingroup tensor
 * \tparam ScalarType Data type of the tensor elements.
 * \param ft Fourth-order tensor .
 * \return Voigt notation matrix.
 *
 * This function converts a fourth-order tensor to a Voigt notation matrix, which is a symmetric 6x6 matrix
 * containing the unique components of the input tensor. The mapping from the tensor indices to the Voigt notation
 * indices is performed by the toVoigt function.
 *
 * \remarks The current implementation
 * does not take advantage of this symmetry.
 */
template <typename ScalarType = double>
Eigen::Matrix<ScalarType, 6, 6> toVoigt(const Eigen::TensorFixedSize<ScalarType, Eigen::Sizes<3, 3, 3, 3>>& ft) {
  Eigen::Matrix<ScalarType, 6, 6> mat;
  for (Eigen::Index i = 0; i < 3; ++i)
    for (Eigen::Index j = 0; j < 3; ++j)
      for (Eigen::Index k = 0; k < 3; ++k)
        for (Eigen::Index l = 0; l < 3; ++l)
          mat(toVoigt(i, j), toVoigt(k, l)) = ft(i, j, k, l);
  return mat;
}

Eigen::Matrix3d generateSPDMatrixWithCommonEigenvalues(double lambda1, double lambda3,double diff) {
    // Step 1: Define the eigenvalues
    // lambda1 = lambda2 (common eigenvalues), lambda3 is distinct
    Eigen::Vector3d eigenvalues(lambda1, lambda1+diff, lambda3);

    // Step 2: Generate a random orthogonal matrix (eigenvectors)
    static Eigen::Matrix3d randomMatrix = Eigen::Matrix3d::Random();
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(randomMatrix);
    Eigen::Matrix3d V = qr.householderQ(); // Orthonormal matrix

    // Step 3: Construct the SPD matrix
    Eigen::Matrix3d Lambda = eigenvalues.asDiagonal(); // Diagonal matrix of eigenvalues
    Eigen::Matrix3d C = V * Lambda * V.transpose();   // SPD matrix

    return C;
}

int main()
{
  Eigen::Matrix3d C;
  C.setRandom();
  C=(C.transpose()+C).eval();
  C+=3*Eigen::Matrix3d::Identity();

for (int i=0; i<17; ++i){
  std::cout<<"i: "<<i<<std::endl;
    C= generateSPDMatrixWithCommonEigenvalues(1.5,7.3,std::pow(10,-i));
     std::cout << C << std::endl;
    // std::cout << principalStretches(C.reshaped()).first << std::endl;

Eigen::Vector<double,6> CVoigt= toVoigt(C);
Eigen::Vector<autodiff::dual2nd,6> CVoigtDual= CVoigt;
DeviatoricMaterial mat;
auto gradientDouble = toVoigt(mat.stresses(CVoigt),false)/2; // dive by two to comare with gradient
    std::cout<<"gr[0]: "<<gradientDouble.transpose()<<std::endl; 

    auto hessianDouble = toVoigt(mat.tangentModuli(CVoigt))/4; // dive by four to comare with hessian

autodiff::dual2nd e;
Eigen::Vector<double,6> g;
Eigen::Matrix<double,6,6> h;
auto lam= [](auto Cvec){ 
  DeviatoricMaterial matD;
  return matD.energy(Cvec);};
 hessian(lam,autodiff::wrt(CVoigtDual),autodiff::at(CVoigtDual),e,g,h);
//gradient(lam,autodiff::wrt(CVoigtDual),autodiff::at(CVoigtDual),e,g);

//  std::cout<<"g[0]: "<<g.transpose()<<std::endl;
  std::cout<<"Diff: "<<(g-gradientDouble).norm()<<std::endl;
    // std::cout<<"hr[0]: "<<hessianDouble<<std::endl; 
  //  std::cout<<"h[0]: "<<h.transpose()<<std::endl;
  std::cout<<"DiffH: "<<(h-hessianDouble).norm()<<std::endl;
// std::cout<<"g[1]: "<<g[1].transpose()<<std::endl;
// std::cout<<"g[2]: "<<g[2].transpose()<<std::endl;

//  std::cout<<"h[0]: "<<h(0,0)<<std::endl;
//  std::cout<<"h[1]: "<<h[1].transpose()<<std::endl;
//  std::cout<<"h[2]: "<<h[2].transpose()<<std::endl;
 }

// auto lam2= [](auto Cvec){ return principalStretchesAD(Cvec).first;};
//   hessianN(lam2,autodiff::wrt(CvecR),autodiff::at(CvecR),e,g,h);

//   std::cout<<"g[0]: "<<g[0].transpose()<<std::endl;
// std::cout<<"g[1]: "<<g[1].transpose()<<std::endl;
// std::cout<<"g[2]: "<<g[2].transpose()<<std::endl;

// std::cout<<"h[0]: "<<h[0].transpose()<<std::endl;
// std::cout<<"h[1]: "<<h[1].transpose()<<std::endl;
// std::cout<<"h[2]: "<<h[2].transpose()<<std::endl;
  return 0; 
}