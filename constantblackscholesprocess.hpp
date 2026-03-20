#pragma once

#include <ql/stochasticprocess.hpp>
#include <ql/types.hpp>

namespace QuantLib {

    class ConstantBlackScholesProcess : public StochasticProcess1D {
      public:
        ConstantBlackScholesProcess(Real s0,
                                    Rate riskFreeRate,
                                    Rate dividendYield,
                                    Volatility volatility);

        Real x0() const override;

        Real drift(Time t, Real x) const override;
        Real diffusion(Time t, Real x) const override;

        Real apply(Real x, Real dx) const override;

        Real expectation(Time t0, Real x0, Time dt) const override;
        Real variance(Time t0, Real x0, Time dt) const override;
        Real stdDeviation(Time t0, Real x0, Time dt) const override;

        Real evolve(Time t0, Real x0, Time dt, Real dw) const override;

        // handy accessors (useful for debugging / benchmarks)
        Rate riskFreeRate() const { return r_; }
        Rate dividendYield() const { return q_; }
        Volatility volatility() const { return sigma_; }

      private:
        Real s0_;
        Rate r_;
        Rate q_;
        Volatility sigma_;
    };

} // namespace QuantLib