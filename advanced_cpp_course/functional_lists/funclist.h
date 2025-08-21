#ifndef FUNCLIST_H
#define FUNCLIST_H

#include <string>
#include <iterator>
#include <functional>
#include <sstream>

namespace flist {

inline const auto empty = []([[maybe_unused]] auto f, auto a) {
    return a;
};

inline const auto cons = [](auto x, auto l) {
    return [=](auto f, auto a) {
        return f(x, l(f, a));
    };
};

namespace detail {
    
    inline auto create_help() {
        return empty;
    }
    
    inline auto create_help(auto x0, auto... xi) {
        return cons(x0, create_help(xi...));
    }
    
    inline std::string rev_string(auto x) {
        std::ostringstream oss;
        oss << x;
        std::string ret = oss.str();
        reverse(ret.begin(), ret.end());
        return ret;
    }
    
    inline auto of_range_calc(const auto &f, const auto& a, auto beg, const auto& end) {
        if (beg == end)
            return a;
        return f(*beg, of_range_calc(f, a, std::next(beg), end));
    };
    
} // detail

inline const auto create = [](auto... xi) {
    return detail::create_help(xi...);
};

inline const auto of_range = [](auto r) {
    auto get_r = [=] {
        if constexpr (std::ranges::bidirectional_range<decltype(r)>)
            return &r;
        else
            return &r.get();
    };
    
    return [=](auto f, auto a) {
        return detail::of_range_calc(f, a, get_r()->begin(), get_r()->end());
    };
};

inline const auto concat = [](auto l, auto k) {
    return [=](auto f, auto a) {
        return l(f, k(f, a));
    };
};

inline const auto rev = [](auto l) {
    return [=](auto f, auto a) {
        using A = decltype(a);
        using F = decltype(f);
        using fun_A_FA = std::function<A(F, A)>;
        fun_A_FA emp = empty;
        return l(
            [=](auto x, fun_A_FA a2) {
                fun_A_FA ret = [=](F f3, A a3) {
                    return a2(f3, f3(x, a3));
                };
                return ret;
            },
            emp
        )(f, a);
    };
};

inline const auto map = [](auto m, auto l) {
    return [=](auto f, auto a) {
        return l(
            [=](auto x, decltype(a) a2) {
                return f(m(x), a2);
            },
            a
        );
    };
};

inline const auto filter = [](auto p, auto l) {
    return [=](auto f, auto a) {
        return l(
            [=](auto x, decltype(a) a2) {
                return p(x) ? f(x, a2) : a2;
            },
            a
        );
    };
};

inline const auto flatten = [](auto l) {
    return [=](auto f, auto a) {
        return l(
            [=](auto x, decltype(a) a2) {
                return x(f, a2);
            },
            a
        );
    };
};

inline const auto as_string = [](const auto& l) {
    std::string ret;
    bool is_empty = true;
    auto f = [&](auto x, [[maybe_unused]] auto a) {
        if (is_empty) {
            ret += detail::rev_string(x);
            is_empty = false;
        }
        else {
            ret += ";" + detail::rev_string(x);
        }
        return 0;   // Signature of lists forces to return something.
    };
    l(f, 0);
    reverse(ret.begin(), ret.end());
    return "[" + ret + "]";
};

} // flist

#endif
