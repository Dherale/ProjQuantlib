/*! \file mceuropeanengine.hpp
    \brief Monte Carlo European option engine
*/

#ifndef montecarlo_european_engine_hpp
#define montecarlo_european_engine_hpp

#include <ql/pricingengines/vanilla/mceuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/exercise.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/timegrid.hpp>

#include "constantprocesshelper.hpp"

namespace QuantLib {

    //! European option pricing engine using Monte Carlo simulation
    /*! \ingroup vanillaengines

        \test the correctness of the returned value is tested by
              checking it against analytic results.
    */
    template <class RNG = PseudoRandom, class S = Statistics>
    class MCEuropeanEngine_2 : public MCVanillaEngine<SingleVariate, RNG, S> {
      public:
        typedef typename MCVanillaEngine<SingleVariate, RNG, S>::path_generator_type
            path_generator_type;
        typedef typename MCVanillaEngine<SingleVariate, RNG, S>::path_pricer_type path_pricer_type;
        typedef typename MCVanillaEngine<SingleVariate, RNG, S>::stats_type stats_type;

        // constructor
        MCEuropeanEngine_2(const boost::shared_ptr<GeneralizedBlackScholesProcess>& process,
                           Size timeSteps,
                           Size timeStepsPerYear,
                           bool brownianBridge,
                           bool antitheticVariate,
                           Size requiredSamples,
                           Real requiredTolerance,
                           Size maxSamples,
                           BigNatural seed,
                           bool constantParameters);

      protected:
        boost::shared_ptr<path_pricer_type> pathPricer() const override;
        boost::shared_ptr<path_generator_type> pathGenerator() const override;

      private:
        bool constantParameters_ = false;
    };

    //! Monte Carlo European engine factory
    template <class RNG = PseudoRandom, class S = Statistics>
    class MakeMCEuropeanEngine_2 {
      public:
        MakeMCEuropeanEngine_2(const boost::shared_ptr<GeneralizedBlackScholesProcess>&);
        // named parameters
        MakeMCEuropeanEngine_2& withSteps(Size steps);
        MakeMCEuropeanEngine_2& withStepsPerYear(Size steps);
        MakeMCEuropeanEngine_2& withBrownianBridge(bool b = true);
        MakeMCEuropeanEngine_2& withSamples(Size samples);
        MakeMCEuropeanEngine_2& withAbsoluteTolerance(Real tolerance);
        MakeMCEuropeanEngine_2& withMaxSamples(Size samples);
        MakeMCEuropeanEngine_2& withSeed(BigNatural seed);
        MakeMCEuropeanEngine_2& withAntitheticVariate(bool b = true);
        MakeMCEuropeanEngine_2& withConstantParameters(bool b = true);
        // conversion to pricing engine
        operator boost::shared_ptr<PricingEngine>() const;

      private:
        boost::shared_ptr<GeneralizedBlackScholesProcess> process_;
        bool antithetic_;
        Size steps_, stepsPerYear_, samples_, maxSamples_;
        Real tolerance_;
        bool brownianBridge_;
        BigNatural seed_;
        bool constantParameters_ = false;
    };


    // inline definitions

    template <class RNG, class S>
    inline MCEuropeanEngine_2<RNG, S>::MCEuropeanEngine_2(
        const boost::shared_ptr<GeneralizedBlackScholesProcess>& process,
        Size timeSteps,
        Size timeStepsPerYear,
        bool brownianBridge,
        bool antitheticVariate,
        Size requiredSamples,
        Real requiredTolerance,
        Size maxSamples,
        BigNatural seed,
        bool constantParameters)
    : MCVanillaEngine<SingleVariate, RNG, S>(process,
                                             timeSteps,
                                             timeStepsPerYear,
                                             brownianBridge,
                                             antitheticVariate,
                                             false,
                                             requiredSamples,
                                             requiredTolerance,
                                             maxSamples,
                                             seed),
      constantParameters_(constantParameters) {}


    template <class RNG, class S>
    inline boost::shared_ptr<typename MCEuropeanEngine_2<RNG, S>::path_pricer_type>
    MCEuropeanEngine_2<RNG, S>::pathPricer() const {

        boost::shared_ptr<PlainVanillaPayoff> payoff =
            boost::dynamic_pointer_cast<PlainVanillaPayoff>(this->arguments_.payoff);
        QL_REQUIRE(payoff, "non-plain payoff given");

        boost::shared_ptr<GeneralizedBlackScholesProcess> process =
            boost::dynamic_pointer_cast<GeneralizedBlackScholesProcess>(this->process_);
        QL_REQUIRE(process, "Black-Scholes process required");

        return boost::shared_ptr<typename MCEuropeanEngine_2<RNG, S>::path_pricer_type>(
            new EuropeanPathPricer(payoff->optionType(),
                                   payoff->strike(),
                                   process->riskFreeRate()->discount(this->timeGrid().back())));
    }

    template <class RNG, class S>
    inline boost::shared_ptr<typename MCEuropeanEngine_2<RNG, S>::path_generator_type>
    MCEuropeanEngine_2<RNG, S>::pathGenerator() const {

        TimeGrid grid = this->timeGrid();
        typename RNG::rsg_type gen =
            RNG::make_sequence_generator(grid.size() - 1, this->seed_);

        if (!constantParameters_) {
            return boost::shared_ptr<path_generator_type>(
                new path_generator_type(this->process_, grid, gen, this->brownianBridge_));
        }

        boost::shared_ptr<EuropeanExercise> exercise =
            boost::dynamic_pointer_cast<EuropeanExercise>(this->arguments_.exercise);
        QL_REQUIRE(exercise, "wrong exercise given");

        boost::shared_ptr<GeneralizedBlackScholesProcess> process =
            boost::dynamic_pointer_cast<GeneralizedBlackScholesProcess>(this->process_);
        QL_REQUIRE(process, "Black-Scholes process required");

        boost::shared_ptr<StochasticProcess1D> constProcess =
            makeConstantBlackScholesProcess(process, exercise->lastDate());

        return boost::shared_ptr<path_generator_type>(
            new path_generator_type(constProcess, grid, gen, this->brownianBridge_));
    }


    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>::MakeMCEuropeanEngine_2(
        const boost::shared_ptr<GeneralizedBlackScholesProcess>& process)
    : process_(process), antithetic_(false), steps_(Null<Size>()), stepsPerYear_(Null<Size>()),
      samples_(Null<Size>()), maxSamples_(Null<Size>()), tolerance_(Null<Real>()),
      brownianBridge_(false), seed_(0), constantParameters_(false) {}

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>& MakeMCEuropeanEngine_2<RNG, S>::withSteps(Size steps) {
        steps_ = steps;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withStepsPerYear(Size steps) {
        stepsPerYear_ = steps;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withSamples(Size samples) {
        QL_REQUIRE(tolerance_ == Null<Real>(), "tolerance already set");
        samples_ = samples;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withAbsoluteTolerance(Real tolerance) {
        QL_REQUIRE(samples_ == Null<Size>(), "number of samples already set");
        QL_REQUIRE(RNG::allowsErrorEstimate,
                   "chosen random generator policy "
                   "does not allow an error estimate");
        tolerance_ = tolerance;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withMaxSamples(Size samples) {
        maxSamples_ = samples;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withSeed(BigNatural seed) {
        seed_ = seed;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withBrownianBridge(bool brownianBridge) {
        brownianBridge_ = brownianBridge;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withAntitheticVariate(bool b) {
        antithetic_ = b;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>&
    MakeMCEuropeanEngine_2<RNG, S>::withConstantParameters(bool b) {
        constantParameters_ = b;
        return *this;
    }

    template <class RNG, class S>
    inline MakeMCEuropeanEngine_2<RNG, S>::operator boost::shared_ptr<PricingEngine>() const {
        QL_REQUIRE(steps_ != Null<Size>() || stepsPerYear_ != Null<Size>(),
                   "number of steps not given");
        QL_REQUIRE(steps_ == Null<Size>() || stepsPerYear_ == Null<Size>(),
                   "number of steps overspecified");

        return boost::shared_ptr<PricingEngine>(new MCEuropeanEngine_2<RNG, S>(process_,
                                                                               steps_,
                                                                               stepsPerYear_,
                                                                               brownianBridge_,
                                                                               antithetic_,
                                                                               samples_,
                                                                               tolerance_,
                                                                               maxSamples_,
                                                                               seed_,
                                                                               constantParameters_));
    }

}

#endif