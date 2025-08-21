# Functional Lists

## Task

The goal of the task is to implement operations on lists that implicitly
remember a sequence of elements <code>[x<sub>0</sub>, …, x<sub>n−1</sub>]</code>.

For a type `L` representing such a list, we only require it to have a
template method with the following signature:

``` cpp
template <typename F, typename A> A operator()(F f, A a);
```

where type `F` has a template method with the following signature:

``` cpp
template <typename X> A operator()(X x, A a);
```

The only way to operate on a list `l` is to run it as a function with an
appropriate function `f` and an accumulator `a`. For objects `l`, `f`,
`a` of classes `L`, `F`, `A` respectively, and objects <code>x<sub>0</sub></code>,
<code>x<sub>1</sub></code>, ..., <code>x<sub>n-1</sub></code> of classes
<code>X<sub>0</sub></code>, <code>X<sub>1</sub></code>, ..., <code>X<sub>n-1</sub></code>
respectively, the following equality holds:

<pre><code>l(f,a) = f(x<sub>0</sub>, f(x<sub>1</sub>, …, f(x<sub>n−1</sub>, a), …))</code></pre>


where the function calls to `f` on the right-hand side are the
appropriate specializations of `operator()` for type `F`.


## Operations as Objects

All operations listed here should be constant objects with an
appropriately overloaded template method `operator()` to allow them to
be passed as function arguments if necessary.

-   `auto empty`: a constant function representing an empty list
-   `auto cons(auto x, auto l)`: a function returning the list `l` with
    `x` added to its beginning
-   `auto create(auto...)`: a function returning a list consisting of
    the provided arguments
-   `auto of_range(auto r)`: a function returning a list created from
    the elements of `r`; it can be assumed that `r` is of a type
    satisfying the concept `std::ranges::bidirectional_range` or
    optionally wrapped in `std::reference_wrapper`
-   `auto concat(auto l, auto k)`: a function returning a list created
    by concatenating lists `l` and `k`
-   `auto rev(auto l)`: a function returning the list `l` with its
    elements in reverse order
-   `auto map(auto m, auto l)`: a function returning a list derived from
    `l` such that each element `x` is replaced by `m(x)`
-   `auto filter(auto p, auto l)`: a function returning a list derived
    from `l` by keeping only those elements `x` that satisfy the
    predicate `p(x)`
-   `auto flatten(auto l)`: a function returning a list created by
    concatenating the lists stored in the list of lists `l`
-   `std::string as_string(const auto& l)`: a function returning the
    representation of list `l` as a `std::string`, assuming that for
    each element `x` in the list, `os << x` works, where `os` is a
    `basic_ostream`-derived object; see usage examples

All definitions of the above objects should be placed in the `flist`
namespace.
