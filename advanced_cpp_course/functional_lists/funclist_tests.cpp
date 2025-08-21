#include "funclist.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <iostream>
#include <array>
#include <cassert>
#include <cctype>
#include <functional>
#include <list>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    template <typename... Args>
    auto create_ref(const Args&... args) {
        return flist::create(std::ref(args)...);
    }

    auto to_upper = [](auto s) {
        std::string str = s;
        for (char& c : str) c = std::toupper(c);
        return str;
    };
}

int main() {
    // Basic list operations
    auto sum = [](auto x, auto a) { return x + a; };
    auto product = [](auto x, auto a) { return x * a; };
    
    auto empty_list = flist::empty;
    assert(flist::as_string(empty_list) == "[]");
    assert(empty_list(sum, 0) == 0);
    
    auto single_list = flist::cons(42, flist::empty);
    assert(flist::as_string(single_list) == "[42]");
    assert(single_list(sum, 0) == 42);
    
    auto mixed_list = flist::create(1, 2.5, 'a', std::string("test"));
    assert(flist::as_string(mixed_list) == "[1;2.5;a;test]");

    // Range-based list creation
    std::vector vec{10, 20, 30};
    auto range_list = flist::of_range(vec);
    assert(flist::as_string(range_list) == "[10;20;30]");
    assert(range_list(sum, 0) == 60);
    
    std::array arr{1.1, 2.2, 3.3};
    auto array_list = flist::of_range(arr);
    assert(array_list(product, 1.0) == 1.1 * 2.2 * 3.3);

    // List transformations
    auto rev_list = flist::rev(mixed_list);
    assert(flist::as_string(rev_list) == "[test;a;2.5;1]");
    
    auto concat_list = flist::concat(single_list, range_list);
    assert(flist::as_string(concat_list) == "[42;10;20;30]");
    assert(concat_list(sum, 0) == 102);

    // Map and filter operations
    auto square = [](auto x) { return x * x; };
    auto squared_list = flist::map(square, range_list);
    assert(flist::as_string(squared_list) == "[100;400;900]");
    assert(squared_list(sum, 0) == 1400);

    auto is_even = [](auto x) { return x % 2 == 0; };
    auto filtered_list = flist::filter(is_even, concat_list);
    assert(flist::as_string(filtered_list) == "[42;10;20;30]");
    assert(filtered_list(sum, 0) == 102);

    // Nested list operations
    auto list_of_lists = flist::create(
        flist::create(1, 2, 3),
        flist::create(4.4, 5.5),
        flist::create(std::string("apple"), std::string("banana"))
    );
    
    auto flattened = flist::flatten(list_of_lists);
    assert(flist::as_string(flattened) == "[1;2;3;4.4;5.5;apple;banana]");
    
    auto rev_flattened = flist::rev(flattened);
    assert(flist::as_string(rev_flattened) == "[banana;apple;5.5;4.4;3;2;1]");

    // String transformations
    auto str_list = flist::create("hello", "world", "functional", "lists");
    auto upper_list = flist::map(to_upper, str_list);
    assert(flist::rev(upper_list)([](auto s, auto a) { return a + " " + s; },
                         std::string("")) == " HELLO WORLD FUNCTIONAL LISTS");

    // Custom list structure
    auto custom_list = [](auto f, auto a) {
        return f(100, f(200, f(300, a)));
    };
    assert(flist::as_string(custom_list) == "[100;200;300]");
    assert(flist::as_string(flist::rev(custom_list)) == "[300;200;100]");
    assert(flist::as_string(flist::map([](int x) { return x/10; }, custom_list))
                                                                == "[10;20;30]");

    // Edge cases
    auto empty_flatten = flist::flatten(flist::create());
    assert(flist::as_string(empty_flatten) == "[]");
    
    auto null_filter = flist::filter([](auto) { return false; }, range_list);
    assert(flist::as_string(null_filter) == "[]");
    assert(null_filter(sum, 0) == 0);
    
    auto nested_empty = create_ref(flist::empty, flist::empty);
    assert(flist::as_string(flist::flatten(nested_empty)) == "[]");

    // Bidirectional range test
    std::list<int> bidir_range{9, 8, 7};
    auto bidir_list = flist::of_range(bidir_range);
    assert(flist::as_string(bidir_list) == "[9;8;7]");
    assert(flist::as_string(flist::rev(bidir_list)) == "[7;8;9]");

    // Type preservation
    auto char_list = flist::create('a', 'b', 'c');
    auto mapped_char = flist::map([](char c) { return c - 32; }, char_list);
    assert(flist::rev(mapped_char)([](char c, std::string s) { return s + c; },
                                                    std::string("")) == "ABC");

    std::cout << "All tests passed!" << std::endl;
    return 0;
}