#include "constantblackscholesprocess.hpp"

#include <ql/errors.hpp>
#include <cmath>

namespace QuantLib {

    ConstantBlackScholesProcess::ConstantBlackScholesProcess(Real s0,
                                                             Rate riskFreeRate,
                                                             Rate dividendYield,
                                                             Volatility volatility)
    : s0_(s0), r_(riskFreeRate), q_(dividendYield), sigma_(volatility) {
        QL_REQUIRE(s0_ > 0.0, "ConstantBlackScholesProcess: s0 must be > 0");
        QL_REQUIRE(sigma_ >= 0.0, "ConstantBlackScholesProcess: sigma must be >= 0");
    }

    Real ConstantBlackScholesProcess::x0() const {
        return s0_;
    }

    Real ConstantBlackScholesProcess::drift(Time /*t*/, Real x) const {
        // Risk-neutral drift: (r - q) * S
        return (r_ - q_) * x;
    }

    Real ConstantBlackScholesProcess::diffusion(Time /*t*/, Real x) const {
        // Diffusion: sigma * S
        return sigma_ * x;
    }

    Real ConstantBlackScholesProcess::apply(Real x, Real dx) const {
        return x + dx;
    }

    Real ConstantBlackScholesProcess::expectation(Time /*t0*/, Real x0, Time dt) const {
        QL_REQUIRE(dt >= 0.0, "ConstantBlackScholesProcess: dt must be >= 0");
        // E[S_{t+dt}|S_t=x0] = x0 * exp((r-q)dt)
        return x0 * std::exp((r_ - q_) * dt);
    }

    Real ConstantBlackScholesProcess::variance(Time /*t0*/, Real x0, Time dt) const {
        QL_REQUIRE(dt >= 0.0, "ConstantBlackScholesProcess: dt must be >= 0");
        if (dt == 0.0 || sigma_ == 0.0)
            return 0.0;

        // Var[S_{t+dt}|S_t=x0] = (x0*exp((r-q)dt))^2 * (exp(sigma^2 dt)-1)
        const Real m = std::exp((r_ - q_) * dt);
        const Real v = std::exp(sigma_ * sigma_ * dt) - 1.0;
        const Real mean = x0 * m;
        return mean * mean * v;
    }

    Real ConstantBlackScholesProcess::stdDeviation(Time t0, Real x0, Time dt) const {
        return std::sqrt(variance(t0, x0, dt));
    }

    Real ConstantBlackScholesProcess::evolve(Time /*t0*/, Real x0, Time dt, Real dw) const {
        QL_REQUIRE(dt >= 0.0, "ConstantBlackScholesProcess: dt must be >= 0");
        if (dt == 0.0)
            return x0;

        if (sigma_ == 0.0) {
            // Deterministic evolution
            return x0 * std::exp((r_ - q_) * dt);
        }

        // Exact GBM step:
        // S_{t+dt} = S_t * exp((r-q-0.5 sigma^2)dt + sigma*sqrt(dt)*Z)
        const Real driftTerm = (r_ - q_ - 0.5 * sigma_ * sigma_) * dt;
        const Real diffTerm  = sigma_ * std::sqrt(dt) * dw; // dw ~ N(0,1)
        return x0 * std::exp(driftTerm + diffTerm);
    }

} 