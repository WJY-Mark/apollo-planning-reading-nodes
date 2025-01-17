/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/**
 * @file: polynomial_xd.cc
 **/

#include "modules/planning/math/polynomial_xd.h"

#include "modules/common/log.h"

namespace apollo {
namespace planning {
// 多项式
PolynomialXd::PolynomialXd(const std::uint32_t order)
    : params_(order + 1, 0.0) {
  CHECK_GE(order, 0);
}

PolynomialXd::PolynomialXd(const std::vector<double>& params)
    : params_(params) {
  CHECK(!params.empty());
}

std::uint32_t PolynomialXd::order() const { return params_.size() - 1; }

void PolynomialXd::SetParams(const std::vector<double>& params) {
  CHECK(!params.empty());
  params_ = params;
}

const std::vector<double>& PolynomialXd::params() const { return params_; }

PolynomialXd PolynomialXd::DerivedFrom(const PolynomialXd& base) {//求导构造函数（如五次多项式求导得到四次多项式） 
  std::vector<double> params;
  if (base.order() <= 0) {
    params.clear();
  } else {
    params.resize(base.params().size() - 1);
    for (std::uint32_t i = 1; i < base.order() + 1; ++i) {
      params[i - 1] = base[i] * i;
    }
  }
  return PolynomialXd(params);
}

PolynomialXd PolynomialXd::IntegratedFrom(const PolynomialXd& base,
                                          const double intercept) {//积分构造函数
  std::vector<double> params;//通过一个多项式的积分来构造另一个多项式（如四次多项式积分变成五次多项式）
  params.resize(base.params().size() + 1);
  params[0] = intercept;
  for (std::uint32_t i = 0; i < base.params().size(); ++i) {
    params[i + 1] = base[i] / (i + 1);
  }
  return PolynomialXd(params);
}

double PolynomialXd::operator()(const double value) const {//反悔多项式在value处的值
  double result = 0.0;
  for (auto rit = params_.rbegin(); rit != params_.rend(); ++rit) {
    result *= value;
    result += (*rit);
  }//  ((((a5x+a4)x+a3)x+a2)x+a1)x+a0
  return result;
}

double PolynomialXd::operator[](const std::uint32_t index) const {
  if (index >= params_.size()) {
    return 0.0;
  } else {
    return params_[index];
  }
}

}  // namespace planning
}  // namespace apollo
