#pragma once

#include <string>

#include "tensor.hpp"

namespace autogradpp {
    /// Returns the library version as a string.
    inline std::string version() {
        return "0.1.0";
    }
}
