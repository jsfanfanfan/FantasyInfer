#include "runtime/runtime_op.hpp"
#include "data/tensor_utils.hpp"

namespace fantasy_infer {
RuntimeOperator::~RuntimeOperator() {
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }
}

}  // namespace fantasy_infer
