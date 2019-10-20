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
 * @file
 **/

#ifndef MODULES_PLANNING_TASKS_DP_POLY_PATH_COMPARABLE_COST_H_
#define MODULES_PLANNING_TASKS_DP_POLY_PATH_COMPARABLE_COST_H_

#include <cmath>
#include <cstdlib>

#include <array>
#include <vector>

namespace apollo {
namespace planning {
// 代价函数封装成一个类
class ComparableCost {
 public:
  ComparableCost() = default;   // 默认的构造函数
  ComparableCost(const bool has_collision, const bool out_of_boundary,  // 是否有碰撞, 是否超出了边界
                 const bool out_of_lane, const float safety_cost_,      // 是否超出了lane(车道), 安全代价的值
                 const float smoothness_cost_)                          // 平滑代价的值                       
      : safety_cost(safety_cost_), smoothness_cost(smoothness_cost_) {
    cost_items[HAS_COLLISION] = has_collision;
    cost_items[OUT_OF_BOUNDARY] = out_of_boundary;
    cost_items[OUT_OF_LANE] = out_of_lane;
  }
  ComparableCost(const ComparableCost &) = default;                     // copy构造函数  

  int CompareTo(const ComparableCost &other) const {                    // 深度拷贝
    for (size_t i = 0; i < cost_items.size(); ++i) {
      if (cost_items[i]) {
        if (other.cost_items[i]) {
          continue;
        } else {
          return 1;
        }
      } else {
        if (other.cost_items[i]) {
          return -1;
        } else {
          continue;
        }
      }
    }

    constexpr float kEpsilon = 1e-12;                                  // 迭代代价值
    const float diff = safety_cost + smoothness_cost - other.safety_cost -
                       other.smoothness_cost;
    if (std::fabs(diff) < kEpsilon) {
      return 0;
    } else if (diff > 0) {
      return 1;
    } else {
      return -1;
    }
  }                                                                    // 重载各种操作符
  ComparableCost operator+(const ComparableCost &other) {          // ComparableCost相加  +实际也用了+=
    ComparableCost lhs = *this;
    lhs += other;
    return lhs;
  }
  ComparableCost &operator+=(const ComparableCost &other) {        //!!注意 此类相加时，不仅cost相加 其cost_items逻辑序列也按或逻辑继承
    for (size_t i = 0; i < cost_items.size(); ++i) {               //原因在于例如轨迹上一个点发生了碰撞，则整条轨迹都相当于发生了碰撞
      cost_items[i] = (cost_items[i] || other.cost_items[i]);      //碰撞这条逻辑应当被继承下去，并用于最终比较各条轨迹ComparableCost的大小
    }
    safety_cost += other.safety_cost;
    smoothness_cost += other.smoothness_cost;
    return *this;
  }
  bool operator>(const ComparableCost &other) const {
    return this->CompareTo(other) > 0;
  }
  bool operator>=(const ComparableCost &other) const {
    return this->CompareTo(other) >= 0;
  }
  bool operator<(const ComparableCost &other) const {
    return this->CompareTo(other) < 0;
  }
  bool operator<=(const ComparableCost &other) const {
    return this->CompareTo(other) <= 0;
  }
  /*
   * cost_items represents an array of factors that affect the cost,
   * The level is from most critical to less critical.
   * It includes:
   * (0) has_collision or out_of_boundary
   * (1) out_of_lane
   *
   * NOTICE: Items could have same critical levels
   */
  static const size_t HAS_COLLISION = 0;   //ComparableCost有三个优先级状态，优先级从高到低，比较大小时，先比较这些状态，优先级一样才比较cost值。
  static const size_t OUT_OF_BOUNDARY = 1;
  static const size_t OUT_OF_LANE = 2;
  std::array<bool, 3> cost_items = {{false, false, false}};      // 代价函数的条目, 目前有三个代价函数(是否碰撞, 是否超出了边界, 是否超出了车道)

  // cost from distance to obstacles or boundaries
  float safety_cost = 0.0f;                                      // 到障碍物或者是边界的距离 成本代价
  // cost from deviation from lane center, path curvature etc
  float smoothness_cost = 0.0f;                                  // 偏离车道中心线或者是车道曲率的 成本代价
};

}  // namespace planning
}  // namespace apollo

#endif  // MODULES_PLANNING_TASKS_DP_POLY_PATH_COMPARABLE_COST_H_
