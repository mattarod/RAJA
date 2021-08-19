#ifndef FORALL_PARAM_HPP
#define FORALL_PARAM_HPP

// Used in omp reduction
int use_dev = 1;

namespace detail
{

  //
  //
  // Invoke Forall with Params.
  //
  //
  CAMP_SUPPRESS_HD_WARN
  template <typename Fn,
            camp::idx_t... Sequence,
            typename Params,
            typename... Ts>
  CAMP_HOST_DEVICE constexpr auto invoke_with_order(Params&& params,
                                                    Fn&& f,
                                                    camp::idx_seq<Sequence...>,
                                                    Ts&&... extra)
  {
    return f(extra..., ( params.template get_lambda_args<Sequence>() )...);
  }

  CAMP_SUPPRESS_HD_WARN
  template <typename Params, typename Fn, typename... Ts>
  CAMP_HOST_DEVICE constexpr auto invoke(Params&& params, Fn&& f, Ts&&... extra)
  {
    return invoke_with_order(
        camp::forward<Params>(params),
        camp::forward<Fn>(f),
        typename camp::decay<Params>::lambda_params_seq(),
        camp::forward<Ts>(extra)...);
  }



  //
  //
  // Forall Parameter Packing type
  //
  //
  template<typename... Params>
  struct ForallParamPack {
    using Base = camp::tuple<Params...>;
    using params_seq = camp::make_idx_seq_t< camp::tuple_size<Base>::value >;
    Base param_tup;

  private:

    template<camp::idx_t Seq>
    constexpr auto lambda_args( camp::idx_seq<Seq> )
        -> decltype(
             camp::get<Seq>(param_tup).get_lambda_arg_tup()
           )
    {
      return camp::get<Seq>(param_tup).get_lambda_arg_tup();
    }

    template<camp::idx_t First, camp::idx_t... Seq>
    constexpr auto lambda_args( camp::idx_seq<First, Seq...> )
        -> decltype(
             camp::tuple_cat_pair(
               camp::get<First>(param_tup).get_lambda_arg_tup(),
               lambda_args(camp::idx_seq<Seq...>())
             )
           )
    {
      return camp::tuple_cat_pair(
               camp::get<First>(param_tup).get_lambda_arg_tup(),
               lambda_args(camp::idx_seq<Seq...>())
             );
    }

    // Init
    template<typename EXEC_POL, camp::idx_t... Seq, typename ...Args>
    static void constexpr detail_init(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params, Args&& ...args) {
      CAMP_EXPAND(init<EXEC_POL>( camp::get<Seq>(f_params.param_tup), std::forward<Args>(args)... ));
    }

    // Combine
    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& out, const ForallParamPack& in ) {
      CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(out.param_tup), camp::get<Seq>(in.param_tup)));
    }

    template<typename EXEC_POL, camp::idx_t... Seq>
    RAJA_HOST_DEVICE
    static void constexpr detail_combine(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(combine<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }
    
    // Resolve
    template<typename EXEC_POL, camp::idx_t... Seq>
    static void constexpr detail_resolve(EXEC_POL, camp::idx_seq<Seq...>, ForallParamPack& f_params ) {
      CAMP_EXPAND(resolve<EXEC_POL>( camp::get<Seq>(f_params.param_tup) ));
    }

    template<typename Last>
    static size_t constexpr count_lambda_args() { return Last::num_lambda_args; }
    template<typename First, typename Second, typename... Rest>
    static size_t constexpr count_lambda_args() { return First::num_lambda_args + count_lambda_args<Second, Rest...>(); }

  public:
    ForallParamPack (){}
    ForallParamPack(Params... params) {
      param_tup = camp::make_tuple(params...);
    };

    using lambda_params_seq = camp::make_idx_seq_t<count_lambda_args<Params...>()>;

    template<camp::idx_t Idx>
    constexpr auto get_lambda_args()
        -> decltype(  *camp::get<Idx>( lambda_args(params_seq{}) )  ) {
      return (  *camp::get<Idx>( lambda_args(params_seq{}) )  );
    }

    // Init
    template<typename EXEC_POL, typename ...Args>
    friend void constexpr init( ForallParamPack& f_params, Args&& ...args) {
      detail_init(EXEC_POL(), params_seq{}, f_params, std::forward<Args>(args)... );
    }

    // Combine
    template<typename EXEC_POL, typename ...Args>
    RAJA_HOST_DEVICE
    friend void constexpr combine(ForallParamPack& f_params, Args&& ...args) {
      detail_combine(EXEC_POL(), params_seq{}, f_params, std::forward<Args>(args)... );
    }

    // Resolve
    template<typename EXEC_POL, typename ...Args>
    friend void constexpr resolve( ForallParamPack& f_params, Args&& ...args) {
      detail_resolve(EXEC_POL(), params_seq{}, f_params , std::forward<Args>(args)... );
    }
  };


} //  namespace detail

#include "sequential/forall.hpp"
#include "openmp/forall.hpp"
#include "omp-target/forall.hpp"
#include "cuda/forall.hpp"
#include "hip/forall.hpp"

template<typename ExecPol, typename B, typename... Params>
void forall_param(int N, const B& body, Params... params) {
  detail::forall_param(ExecPol(), N, body, params...);
}

#endif //  FORALL_PARAM_HPP
