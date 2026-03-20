#ifndef constant_process_helper_hpp
#define constant_process_helper_hpp

#include <ql/processes/blackscholesprocess.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/types.hpp>
#include <ql/time/date.hpp>

#include "constantblackscholesprocess.hpp"

namespace QuantLib {

    struct ConstantBlackScholesParameters {
        Real s0;
        Rate r;
        Rate q;
        Volatility sigma;
    };

    inline ConstantBlackScholesParameters extractConstantBlackScholesParameters(
        const GeneralizedBlackScholesProcess& process,
        const Date& maturity) {

        ConstantBlackScholesParameters params;

        params.s0 = process.x0();

        params.r =
            process.riskFreeRate()
                ->zeroRate(maturity,
                           process.riskFreeRate()->dayCounter(),
                           Continuous,
                           NoFrequency)
                .rate();

        params.q =
            process.dividendYield()
                ->zeroRate(maturity,
                           process.dividendYield()->dayCounter(),
                           Continuous,
                           NoFrequency)
                .rate();

        params.sigma =
            process.blackVolatility()->blackVol(maturity, params.s0);

        return params;
    }

    inline ext::shared_ptr<StochasticProcess1D> makeConstantBlackScholesProcess(
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
        const Date& maturity) {

        QL_REQUIRE(process, "null Black-Scholes process");

        ConstantBlackScholesParameters params =
            extractConstantBlackScholesParameters(*process, maturity);

        return ext::shared_ptr<StochasticProcess1D>(
            new ConstantBlackScholesProcess(
                params.s0, params.r, params.q, params.sigma));
    }

}

#endif