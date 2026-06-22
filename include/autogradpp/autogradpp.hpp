#pragma once

#include <string>

#include "tensor.hpp"
#include "node.hpp"
#include "operands.hpp"
#include "mlp.hpp"

namespace autogradpp {
    /// Returns the library version as a string.
    inline std::string version() {
        return "0.1.0";
    }
}
