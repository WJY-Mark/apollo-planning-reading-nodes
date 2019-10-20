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

#include "modules/planning/toolkits/optimizers/dp_poly_path/trajectory_cost.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "modules/common/proto/pnc_point.pb.h"

#include "modules/common/configs/vehicle_config_helper.h"
#include "modules/common/math/vec2d.h"
#include "modules/common/util/util.h"
#include "modules/planning/common/planning_gflags.h"

namespace apollo {
namespace planning {

using apollo::common::TrajectoryPoint;
using apollo::common::math::Box2d;
using apollo::common::math::Sigmoid;
using apollo::common::math::Vec2d;

TrajectoryCost::TrajectoryCost(       //初始化，除各变量赋值以外，还包括计算自车sl边界，静态障碍物sl边界，动态障碍物sl边界序列。
    const DpPolyPathConfig &config, const ReferenceLine &reference_line,
    const bool is_change_lane_path,
    const std::vector<const PathObstacle *> &obstacles,
    const common::VehicleParam &vehicle_param,
    const SpeedData &heuristic_speed_data, const common::SLPoint &init_sl_point)
    : config_(config),
      reference_line_(&reference_line),
      is_change_lane_path_(is_change_lane_path),
      vehicle_param_(vehicle_param),
      heuristic_speed_data_(heuristic_speed_data),
      init_sl_point_(init_sl_point) {
  const float total_time =
      std::min(heuristic_speed_data_.TotalTime(), FLAGS_prediction_total_time);

  num_of_time_stamps_ = static_cast<uint32_t>(
      std::floor(total_time / config.eval_time_interval()));

  for (const auto *ptr_path_obstacle : obstacles) {
    if (ptr_path_obstacle->IsIgnore()) {
      continue;
    } else if (ptr_path_obstacle->LongitudinalDecision().has_stop()) {
      continue;
    }
    const auto &sl_boundary = ptr_path_obstacle->PerceptionSLBoundary();

    const float adc_left_l =
        init_sl_point_.l() + vehicle_param_.left_edge_to_center();
    const float adc_right_l =
        init_sl_point_.l() - vehicle_param_.right_edge_to_center();

    if (adc_left_l + FLAGS_lateral_ignore_buffer < sl_boundary.start_l() ||
        adc_right_l - FLAGS_lateral_ignore_buffer > sl_boundary.end_l()) {
      continue;
    }

    const auto *ptr_obstacle = ptr_path_obstacle->obstacle();
    bool is_bycycle_or_pedestrian =
        (ptr_obstacle->Perception().type() ==
             perception::PerceptionObstacle::BICYCLE ||
         ptr_obstacle->Perception().type() ==
             perception::PerceptionObstacle::PEDESTRIAN);

    if (Obstacle::IsVirtualObstacle(ptr_obstacle->Perception())) {
      // Virtual obstacle
      continue;
    } else if (Obstacle::IsStaticObstacle(ptr_obstacle->Perception()) ||
               is_bycycle_or_pedestrian) {
      static_obstacle_sl_boundaries_.push_back(std::move(sl_boundary));  //静态障碍物 根据sl坐标系下的boundary求cost。
    } else {
      std::vector<Box2d> box_by_time;
      for (uint32_t t = 0; t <= num_of_time_stamps_; ++t) {
        TrajectoryPoint trajectory_point =
            ptr_obstacle->GetPointAtTime(t * config.eval_time_interval());

        Box2d obstacle_box = ptr_obstacle->GetBoundingBox(trajectory_point);//动态障碍物 根据全局坐标系下的boundingbox求cost
        constexpr float kBuff = 0.5;
        Box2d expanded_obstacle_box =
            Box2d(obstacle_box.center(), obstacle_box.heading(),
                  obstacle_box.length() + kBuff, obstacle_box.width() + kBuff);  //障碍物膨胀
        box_by_time.push_back(expanded_obstacle_box); //动态障碍物box的时间序列
      }
      dynamic_obstacle_boxes_.push_back(std::move(box_by_time));//动态障碍物
    }
  }
}

ComparableCost TrajectoryCost::CalculatePathCost(
    const QuinticPolynomialCurve1d &curve, const float start_s,
    const float end_s, const uint32_t curr_level, const uint32_t total_level) {
  ComparableCost cost;
  float path_cost = 0.0;
  std::function<float(const float)> quasi_softmax = [this](const float x) {
    const float l0 = this->config_.path_l_cost_param_l0();
    const float b = this->config_.path_l_cost_param_b();
    const float k = this->config_.path_l_cost_param_k();
    return (b + std::exp(-k * (x - l0))) / (1.0 + std::exp(-k * (x - l0)));
  };

  const auto &vehicle_config =
      common::VehicleConfigHelper::instance()->GetConfig();
  const float width = vehicle_config.vehicle_param().width();

  for (float curve_s = 0.0; curve_s < (end_s - start_s);
       curve_s += config_.path_resolution()) {
    const float l = curve.Evaluate(0, curve_s);

    path_cost += l * l * config_.path_l_cost() * quasi_softmax(std::fabs(l));

    double left_width = 0.0;
    double right_width = 0.0;
    reference_line_->GetLaneWidth(curve_s + start_s, &left_width, &right_width);

    constexpr float kBuff = 0.2;
    if (!is_change_lane_path_ && (l + width / 2.0 + kBuff > left_width ||
                                  l - width / 2.0 - kBuff < -right_width)) {
      cost.cost_items[ComparableCost::OUT_OF_BOUNDARY] = true;
    }

    const float dl = std::fabs(curve.Evaluate(1, curve_s));
    path_cost += dl * dl * config_.path_dl_cost();  //对每个离散点计算dl和ddl，并加权求和

    const float ddl = std::fabs(curve.Evaluate(2, curve_s));
    path_cost += ddl * ddl * config_.path_ddl_cost();
  }
  path_cost *= config_.path_resolution();

  if (curr_level == total_level) {
    const float end_l = curve.Evaluate(0, end_s - start_s);
    path_cost +=
        std::sqrt(end_l - init_sl_point_.l() / 2.0) * config_.path_end_l_cost();
  }
  cost.smoothness_cost = path_cost;
  return cost;
}

ComparableCost TrajectoryCost::CalculateStaticObstacleCost(   //注意静态障碍物的cost是在SL坐标系下衡量的
    const QuinticPolynomialCurve1d &curve, const float start_s,
    const float end_s) {
  ComparableCost obstacle_cost;
  for (float curr_s = start_s; curr_s <= end_s;
       curr_s += config_.path_resolution()) {
    const float curr_l = curve.Evaluate(0, curr_s - start_s);
    for (const auto &obs_sl_boundary : static_obstacle_sl_boundaries_) {
      obstacle_cost += GetCostFromObsSL(curr_s, curr_l, obs_sl_boundary);//在SL坐标系（为啥不是全局？）下计算两个boundingbox的cost
    }
  }
  obstacle_cost.safety_cost *= config_.path_resolution();
  return obstacle_cost;
}

ComparableCost TrajectoryCost::CalculateDynamicObstacleCost(  //注意动态障碍物的cost是在XY全局坐标系下衡量的
    const QuinticPolynomialCurve1d &curve, const float start_s,
    const float end_s) const {
  ComparableCost obstacle_cost;
  float time_stamp = 0.0;
  for (size_t index = 0; index < num_of_time_stamps_;
       ++index, time_stamp += config_.eval_time_interval()) {
    common::SpeedPoint speed_point;
    heuristic_speed_data_.EvaluateByTime(time_stamp, &speed_point);
    float ref_s = speed_point.s() + init_sl_point_.s();      //根据启发式速度估计未来每个时间戳自车的s
    if (ref_s < start_s) {
      continue;
    }
    if (ref_s > end_s) {
      break;
    }

    const float s = ref_s - start_s;  // s on spline curve
    const float l = curve.Evaluate(0, s);  //根据五次多项式计算出每个时间戳的sl坐标
    const float dl = curve.Evaluate(1, s);

    const common::SLPoint sl = common::util::MakeSLPoint(ref_s, l);
    const Box2d ego_box = GetBoxFromSLPoint(sl, dl);
    for (const auto &obstacle_trajectory : dynamic_obstacle_boxes_) {
      obstacle_cost +=
          GetCostBetweenObsBoxes(ego_box, obstacle_trajectory.at(index));//根据自车box和动态障碍物box的关系计算cost
    }
  }
  constexpr float kDynamicObsWeight = 1e-6;
  obstacle_cost.safety_cost *=
      (config_.eval_time_interval() * kDynamicObsWeight);
  return obstacle_cost;
}

ComparableCost TrajectoryCost::GetCostFromObsSL(
    const float adc_s, const float adc_l, const SLBoundary &obs_sl_boundary) {
  const auto &vehicle_param =
      common::VehicleConfigHelper::instance()->GetConfig().vehicle_param();

  ComparableCost obstacle_cost;

  const float adc_front_s = adc_s + vehicle_param.front_edge_to_center();
  const float adc_end_s = adc_s - vehicle_param.back_edge_to_center();
  const float adc_left_l = adc_l + vehicle_param.left_edge_to_center();
  const float adc_right_l = adc_l - vehicle_param.right_edge_to_center();

  if (adc_left_l + FLAGS_lateral_ignore_buffer < obs_sl_boundary.start_l() ||
      adc_right_l - FLAGS_lateral_ignore_buffer > obs_sl_boundary.end_l()) {
    return obstacle_cost;
  }

  bool no_overlap = ((adc_front_s < obs_sl_boundary.start_s() ||
                      adc_end_s > obs_sl_boundary.end_s()) ||  // longitudinal
                     (adc_left_l + FLAGS_static_decision_nudge_l_buffer <
                          obs_sl_boundary.start_l() ||
                      adc_right_l - FLAGS_static_decision_nudge_l_buffer >
                          obs_sl_boundary.end_l()));  // lateral

  if (!no_overlap) {
    obstacle_cost.cost_items[ComparableCost::HAS_COLLISION] = true;
  }

  // if obstacle is behind ADC, ignore its cost contribution.
  if (adc_front_s > obs_sl_boundary.end_s()) {
    return obstacle_cost;
  }

  const float delta_l = std::fabs(
      adc_l - (obs_sl_boundary.start_l() + obs_sl_boundary.end_l()) / 2.0);

  const double kSafeDistance = 1.0;
  if (delta_l < kSafeDistance) {
    obstacle_cost.safety_cost +=
        config_.obstacle_collision_cost() *
        Sigmoid(config_.obstacle_collision_distance() - delta_l);
  }

  const float delta_s = std::fabs(
      adc_s - (obs_sl_boundary.start_s() + obs_sl_boundary.end_s()) / 2.0);
  obstacle_cost.safety_cost +=
      config_.obstacle_collision_cost() *
      Sigmoid(config_.obstacle_collision_distance() - delta_s);
  return obstacle_cost;
}

// Simple version: calculate obstacle cost by distance
ComparableCost TrajectoryCost::GetCostBetweenObsBoxes(
    const Box2d &ego_box, const Box2d &obstacle_box) const {
  ComparableCost obstacle_cost;

  const float distance = obstacle_box.DistanceTo(ego_box);
  if (distance > config_.obstacle_ignore_distance()) {
    return obstacle_cost;
  }

  obstacle_cost.safety_cost +=
      config_.obstacle_collision_cost() *
      Sigmoid(config_.obstacle_collision_distance() - distance);
  obstacle_cost.safety_cost +=
      20.0 * Sigmoid(config_.obstacle_risk_distance() - distance);
  return obstacle_cost;
}

Box2d TrajectoryCost::GetBoxFromSLPoint(const common::SLPoint &sl,  //从sl点得到全局坐标系下的box
                                        const float dl) const {
  Vec2d xy_point;
  reference_line_->SLToXY(sl, &xy_point);

  ReferencePoint reference_point = reference_line_->GetReferencePoint(sl.s());

  const float one_minus_kappa_r_d = 1 - reference_point.kappa() * sl.l();  //参考werling附录(6) d'=[1-kr*d]tan(△θ) 
  const float delta_theta = std::atan2(dl, one_minus_kappa_r_d);
  const float theta =
      common::math::NormalizeAngle(delta_theta + reference_point.heading());//全局坐标系下的航向角
  return Box2d(xy_point, theta, vehicle_param_.length(),
               vehicle_param_.width());
}

// TODO(All): optimize obstacle cost calculation time
ComparableCost TrajectoryCost::Calculate(const QuinticPolynomialCurve1d &curve,
                                         const float start_s, const float end_s,
                                         const uint32_t curr_level,
                                         const uint32_t total_level) {
  ComparableCost total_cost;
  // path cost
  total_cost +=
      CalculatePathCost(curve, start_s, end_s, curr_level, total_level);

  // static obstacle cost
  total_cost += CalculateStaticObstacleCost(curve, start_s, end_s);

  // dynamic obstacle cost
  total_cost += CalculateDynamicObstacleCost(curve, start_s, end_s);
  return total_cost;
}

}  // namespace planning
}  // namespace apollo
