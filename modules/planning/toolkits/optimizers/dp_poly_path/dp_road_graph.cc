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
 * @file dp_road_graph.h
 **/

#include "modules/planning/toolkits/optimizers/dp_poly_path/dp_road_graph.h"

#include <algorithm>
#include <utility>

#include "modules/common/configs/vehicle_config_helper.h"
#include "modules/common/log.h"
#include "modules/common/math/cartesian_frenet_conversion.h"
#include "modules/common/proto/error_code.pb.h"
#include "modules/common/proto/pnc_point.pb.h"
#include "modules/common/util/thread_pool.h"
#include "modules/common/util/util.h"

#include "modules/map/hdmap/hdmap_util.h"

#include "modules/planning/common/path/frenet_frame_path.h"
#include "modules/planning/common/planning_context.h"
#include "modules/planning/common/planning_gflags.h"
#include "modules/planning/math/curve1d/quintic_polynomial_curve1d.h"
#include "modules/planning/proto/planning_internal.pb.h"
#include "modules/planning/proto/planning_status.pb.h"

namespace apollo {
namespace planning {

using apollo::common::ErrorCode;                             // 错误码
using apollo::common::SLPoint;                               // sl坐标
using apollo::common::Status;                                // 车辆的状态
using apollo::common::math::CartesianFrenetConverter;        // 笛卡尔坐标系转换为Frenet坐标系
using apollo::common::util::MakeSLPoint;                     // 产生一个SL坐标系的点
using apollo::common::util::ThreadPool;                      // 线程池
// 构造函数
DPRoadGraph::DPRoadGraph(const DpPolyPathConfig &config,     // dp的多项式路径配置
                         const ReferenceLineInfo &reference_line_info, // 参考线的信息
                         const SpeedData &speed_data)        // 速度相关的数据
    : config_(config),
      reference_line_info_(reference_line_info),
      reference_line_(reference_line_info.reference_line()),
      speed_data_(speed_data) {}
// 
bool DPRoadGraph::FindPathTunnel(
    const common::TrajectoryPoint &init_point,               // 轨迹点
    const std::vector<const PathObstacle *> &obstacles,      // 存放障碍物的vector
    PathData *const path_data) {                             // 路径的数据
  CHECK_NOTNULL(path_data);

  init_point_ = init_point;                                  // 起点
  if (!reference_line_.XYToSL(
          {init_point_.path_point().x(), init_point_.path_point().y()}, // 转换为sl坐标系
          &init_sl_point_)) {
    AERROR << "Fail to create init_sl_point from : "
           << init_point.DebugString();
    return false;
  }
  // 转换为frenet坐标系, 主要是有个frenet的frame点
  if (!CalculateFrenetPoint(init_point_, &init_frenet_frame_point_)) {
    AERROR << "Fail to create init_frenet_frame_point_ from : "
           << init_point_.DebugString();
    return false;
  }

  std::vector<DPRoadGraphNode> min_cost_path;   // 最小路径的代价函数 节点的vector
  if (!GenerateMinCostPath(obstacles, &min_cost_path)) {   // 产生最小代价的路径,核心函数，最终输出节点的vector
    AERROR << "Fail to generate graph!";
    return false;
  }
  std::vector<common::FrenetFramePoint> frenet_path;   //根据节点vector的五次多项式进行slice细分（间隔较小）
  float accumulated_s = init_sl_point_.s();
  const float path_resolution = config_.path_resolution();

  for (std::size_t i = 1; i < min_cost_path.size(); ++i) {
    const auto &prev_node = min_cost_path[i - 1];
    const auto &cur_node = min_cost_path[i];

    const float path_length = cur_node.sl_point.s() - prev_node.sl_point.s();
    float current_s = 0.0;
    const auto &curve = cur_node.min_cost_curve;
    while (current_s + path_resolution / 2.0 < path_length) {
      const float l = curve.Evaluate(0, current_s);
      const float dl = curve.Evaluate(1, current_s);
      const float ddl = curve.Evaluate(2, current_s);
      common::FrenetFramePoint frenet_frame_point;
      frenet_frame_point.set_s(accumulated_s + current_s);
      frenet_frame_point.set_l(l);
      frenet_frame_point.set_dl(dl);
      frenet_frame_point.set_ddl(ddl);
      frenet_path.push_back(std::move(frenet_frame_point));
      current_s += path_resolution;
    }
    if (i == min_cost_path.size() - 1) {
      accumulated_s += current_s;
    } else {
      accumulated_s += path_length;
    }
  }
  FrenetFramePath tunnel(frenet_path);
  path_data->SetReferenceLine(&reference_line_);
  path_data->SetFrenetPath(tunnel);
  return true;
}

bool DPRoadGraph::GenerateMinCostPath(                       // 在DP RoadGraph图中的GenerateMinCostPath函数里通过障碍物生成最小代价的路径
    const std::vector<const PathObstacle *> &obstacles,
    std::vector<DPRoadGraphNode> *min_cost_path) {
  CHECK(min_cost_path != nullptr);                           // 做边界检查

  std::vector<std::vector<common::SLPoint>> path_waypoints;  // 二维的路网的点(path_waypoints)
  if (!SamplePathWaypoints(init_point_, &path_waypoints) ||  // 从起点进行数据的采样, 采样的结果保存到二维数组中
      path_waypoints.size() < 1) {
    AERROR << "Fail to sample path waypoints! reference_line_length = "
           << reference_line_.Length();
    return false;
  }
  path_waypoints.insert(path_waypoints.begin(),
                        std::vector<common::SLPoint>{init_sl_point_});//把起点加入进去
  const auto &vehicle_config =
      common::VehicleConfigHelper::instance()->GetConfig();

  TrajectoryCost trajectory_cost(
      config_, reference_line_, reference_line_info_.IsChangeLanePath(),
      obstacles, vehicle_config.vehicle_param(), speed_data_, init_sl_point_);//TrajectoryCost类，用于计算各段五次多项式的cost，
  //注意把speed_data_输入进去了，作为“启发式”。 用于估计自车在未来的位置，从而考虑动态障碍物的cost

  std::list<std::list<DPRoadGraphNode>> graph_nodes;//最终的前向遍历图，类似于神经网络 N个level，每个level一排node。（不含起点）
  graph_nodes.emplace_back();
  graph_nodes.back().emplace_back(init_sl_point_, nullptr, ComparableCost());
  auto &front = graph_nodes.front().front();
  size_t total_level = path_waypoints.size();

  for (std::size_t level = 1; level < path_waypoints.size(); ++level) {//level从1开始，level0是起点init_sl_point_
    const auto &prev_dp_nodes = graph_nodes.back();//auto=std::list<DPRoadGraphNode>
    const auto &level_points = path_waypoints[level];//把采样的SL点拿出来，稍后转化为node auto=std::vector<common::SLPoint>

    graph_nodes.emplace_back();

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < level_points.size(); ++i) {
      const auto &cur_point = level_points[i];

      graph_nodes.back().emplace_back(cur_point, nullptr);//将采样的SLpoint转化为node
      auto &cur_node = graph_nodes.back().back();
      if (FLAGS_enable_multi_thread_in_dp_poly_path) {
        futures.push_back(ThreadPool::pool()->push(std::bind(
            &DPRoadGraph::UpdateNode, this, std::ref(prev_dp_nodes), level,
            total_level, &trajectory_cost, &(front), &(cur_node))));

      } else {
        UpdateNode(prev_dp_nodes, level, total_level, &trajectory_cost, &front,  //进入子函数 这个函数1.完成两层节点间的五次多项式连接
                   &cur_node);                                                   //2.计算这段五次多项式的cost 3选最小的路径+父节点赋给当前node
      }
    }

    for (const auto &f : futures) {
      f.wait();
    }
  }

  // find best path
  DPRoadGraphNode fake_head;  //反向找到最佳路径
  for (const auto &cur_dp_node : graph_nodes.back()) {
    fake_head.UpdateCost(&cur_dp_node, cur_dp_node.min_cost_curve,
                         cur_dp_node.min_cost);
  }

  const auto *min_cost_node = &fake_head;
  while (min_cost_node->min_cost_prev_node) {
    min_cost_node = min_cost_node->min_cost_prev_node;
    min_cost_path->push_back(*min_cost_node);
  }
  if (min_cost_node != &graph_nodes.front().front()) {
    return false;
  }

  std::reverse(min_cost_path->begin(), min_cost_path->end());

  for (const auto &node : *min_cost_path) {
    ADEBUG << "min_cost_path: " << node.sl_point.ShortDebugString();
    planning_debug_->mutable_planning_data()
        ->mutable_dp_poly_graph()
        ->add_min_cost_point()
        ->CopyFrom(node.sl_point);
  }
  return true;
}

void DPRoadGraph::UpdateNode(const std::list<DPRoadGraphNode> &prev_nodes,
                             const uint32_t level, const uint32_t total_level,
                             TrajectoryCost *trajectory_cost,
                             DPRoadGraphNode *front,
                             DPRoadGraphNode *cur_node) {
  DCHECK_NOTNULL(trajectory_cost);
  DCHECK_NOTNULL(front);
  DCHECK_NOTNULL(cur_node);
  for (const auto &prev_dp_node : prev_nodes) {
    const auto &prev_sl_point = prev_dp_node.sl_point;
    const auto &cur_point = cur_node->sl_point;
    float init_dl = 0.0;
    float init_ddl = 0.0;
    if (level == 1) {
      init_dl = init_frenet_frame_point_.dl();
      init_ddl = init_frenet_frame_point_.ddl();
    }
    QuinticPolynomialCurve1d curve(prev_sl_point.l(), init_dl, init_ddl,
                                   cur_point.l(), 0.0, 0.0,
                                   cur_point.s() - prev_sl_point.s());//注意这里的五次多项式输入的是相对坐标，或者说每一段的长度。即cur_s-prev_s
                                                                      //也就是说每个五次多项式都是从s=0开始的。

    if (!IsValidCurve(curve)) {
      continue;
    }
    const auto cost =
        trajectory_cost->Calculate(curve, prev_sl_point.s(), cur_point.s(),
                                   level, total_level) +
        prev_dp_node.min_cost;

    cur_node->UpdateCost(&prev_dp_node, curve, cost);
  }

  // try to connect the current point with the first point directly
  if (level >= 2) {
    const float init_dl = init_frenet_frame_point_.dl();
    const float init_ddl = init_frenet_frame_point_.ddl();
    QuinticPolynomialCurve1d curve(init_sl_point_.l(), init_dl, init_ddl,
                                   cur_node->sl_point.l(), 0.0, 0.0,
                                   cur_node->sl_point.s() - init_sl_point_.s());//五次多项式，六个条件，认为终点侧向速度加速度=0。
    if (!IsValidCurve(curve)) {
      return;
    }
    const auto cost = trajectory_cost->Calculate(
        curve, init_sl_point_.s(), cur_node->sl_point.s(), level, total_level);
    cur_node->UpdateCost(front, curve, cost);
  }
}

bool DPRoadGraph::SamplePathWaypoints(
    const common::TrajectoryPoint &init_point,
    std::vector<std::vector<common::SLPoint>> *const points) {   // 采样的结果放在二维数组中, 每个点是SL的坐标
  CHECK_NOTNULL(points);                                         // 做有效性检查

  const float kMinSampleDistance = 40.0;                         // 最小采样的距离为40
  const float total_length = std::fmin(                          // 获取采样的总长度
      init_sl_point_.s() + std::fmax(init_point.v() * 8.0, kMinSampleDistance),
      reference_line_.Length());                                 // 起点的距离加上未来8秒的路程, 最小采样距离40米, 车道中心参考线的长度, 三者取最大值
  const auto &vehicle_config =
      common::VehicleConfigHelper::instance()->GetConfig();      // VehicleConfigHelper这个单例类的作用就是获取车辆配置的参数, 还有一个重要的功能是计算最大转向半径
  const float half_adc_width = vehicle_config.vehicle_param().width() / 2.0;
  const size_t num_sample_per_level =
      FLAGS_use_navigation_mode ? config_.navigator_sample_num_each_level()
                                : config_.sample_points_num_each_level();

  const bool has_sidepass = HasSidepass();

  constexpr float kSamplePointLookForwardTime = 4.0;
  const float step_length =
      common::math::Clamp(init_point.v() * kSamplePointLookForwardTime,
                          config_.step_length_min(), config_.step_length_max());

  const float level_distance =
      (init_point.v() > FLAGS_max_stop_speed) ? step_length : step_length / 2.0;
  float accumulated_s = init_sl_point_.s();
  float prev_s = accumulated_s;

  auto *status = GetPlanningStatus();
  if (!status->has_pull_over() && status->pull_over().in_pull_over()) {
    status->mutable_pull_over()->set_status(PullOverStatus::IN_OPERATION);
    const auto &start_point = status->pull_over().start_point();
    SLPoint start_point_sl;
    if (!reference_line_.XYToSL(start_point, &start_point_sl)) {
      AERROR << "Fail to change xy to sl.";
      return false;
    }

    if (init_sl_point_.s() > start_point_sl.s()) {
      const auto &stop_point = status->pull_over().stop_point();
      SLPoint stop_point_sl;
      if (!reference_line_.XYToSL(stop_point, &stop_point_sl)) {
        AERROR << "Fail to change xy to sl.";
        return false;
      }
      std::vector<common::SLPoint> level_points(1, stop_point_sl);
      points->emplace_back(level_points);
      return true;
    }
  }

  for (std::size_t i = 0; accumulated_s < total_length; ++i) {
    accumulated_s += level_distance;
    if (accumulated_s + level_distance / 2.0 > total_length) {
      accumulated_s = total_length;
    }
    const float s = std::fmin(accumulated_s, total_length);
    constexpr float kMinAllowedSampleStep = 1.0;
    if (std::fabs(s - prev_s) < kMinAllowedSampleStep) {
      continue;
    }
    prev_s = s;

    double left_width = 0.0;
    double right_width = 0.0;
    reference_line_.GetLaneWidth(s, &left_width, &right_width);

    constexpr float kBoundaryBuff = 0.20;
    const float eff_right_width = right_width - half_adc_width - kBoundaryBuff;
    const float eff_left_width = left_width - half_adc_width - kBoundaryBuff;

    // the heuristic shift of L for lane change scenarios
    const double delta_dl = 1.2 / 20.0;
    const double kChangeLaneDeltaL = common::math::Clamp(
        level_distance * (std::fabs(init_frenet_frame_point_.dl()) + delta_dl),
        1.2, 3.5);

    float kDefaultUnitL = kChangeLaneDeltaL / (num_sample_per_level - 1);
    if (reference_line_info_.IsChangeLanePath() &&
        !reference_line_info_.IsSafeToChangeLane()) {
      kDefaultUnitL = 1.0;
    }
    const float sample_l_range = kDefaultUnitL * (num_sample_per_level - 1);
    float sample_right_boundary = -eff_right_width;
    float sample_left_boundary = eff_left_width;

    const float kLargeDeviationL = 1.75;
    if (reference_line_info_.IsChangeLanePath() ||
        std::fabs(init_sl_point_.l()) > kLargeDeviationL) {
      sample_right_boundary = std::fmin(-eff_right_width, init_sl_point_.l());
      sample_left_boundary = std::fmax(eff_left_width, init_sl_point_.l());

      if (init_sl_point_.l() > eff_left_width) {
        sample_right_boundary = std::fmax(sample_right_boundary,
                                          init_sl_point_.l() - sample_l_range);
      }
      if (init_sl_point_.l() < eff_right_width) {
        sample_left_boundary = std::fmin(sample_left_boundary,
                                         init_sl_point_.l() + sample_l_range);
      }
    }

    std::vector<float> sample_l;
    if (reference_line_info_.IsChangeLanePath() &&
        !reference_line_info_.IsSafeToChangeLane()) {
      sample_l.push_back(reference_line_info_.OffsetToOtherReferenceLine());
    } else if (has_sidepass) {
      // currently only left nudge is supported. Need road hard boundary for
      // both sides
      switch (sidepass_.type()) {
        case ObjectSidePass::LEFT: {
          sample_l.push_back(eff_left_width + config_.sidepass_distance());
          break;
        }
        case ObjectSidePass::RIGHT: {
          sample_l.push_back(-eff_right_width - config_.sidepass_distance());
          break;
        }
        default:
          break;
      }
    } else {
      common::util::uniform_slice(sample_right_boundary, sample_left_boundary,
                                  num_sample_per_level - 1, &sample_l);
    }
    std::vector<common::SLPoint> level_points;
    planning_internal::SampleLayerDebug sample_layer_debug;
    for (size_t j = 0; j < sample_l.size(); ++j) {
      common::SLPoint sl = common::util::MakeSLPoint(s, sample_l[j]);
      sample_layer_debug.add_sl_point()->CopyFrom(sl);
      level_points.push_back(std::move(sl));
    }
    if (!reference_line_info_.IsChangeLanePath() && has_sidepass) {
      auto sl_zero = common::util::MakeSLPoint(s, 0.0);
      sample_layer_debug.add_sl_point()->CopyFrom(sl_zero);
      level_points.push_back(std::move(sl_zero));
    }

    if (!level_points.empty()) {
      planning_debug_->mutable_planning_data()
          ->mutable_dp_poly_graph()
          ->add_sample_layer()
          ->CopyFrom(sample_layer_debug);
      points->emplace_back(level_points);
    }
  }
  return true;
}

bool DPRoadGraph::CalculateFrenetPoint(
    const common::TrajectoryPoint &traj_point,
    common::FrenetFramePoint *const frenet_frame_point) {
  common::SLPoint sl_point;   // 除了转换为sl坐标以外
  if (!reference_line_.XYToSL(
          {traj_point.path_point().x(), traj_point.path_point().y()},
          &sl_point)) {
    return false;
  }
  frenet_frame_point->set_s(sl_point.s());
  frenet_frame_point->set_l(sl_point.l());

  const float theta = traj_point.path_point().theta();    // 还会计算theta值
  const float kappa = traj_point.path_point().kappa();    // 曲率值
  const float l = frenet_frame_point->l();

  ReferencePoint ref_point;                               // 获取道路中心参考线
  ref_point = reference_line_.GetReferencePoint(frenet_frame_point->s());

  const float theta_ref = ref_point.heading();            // 获取道路中心参考线的航向角
  const float kappa_ref = ref_point.kappa();              // 曲率
  const float dkappa_ref = ref_point.dkappa();            // 曲率的导数

  const float dl = CartesianFrenetConverter::CalculateLateralDerivative(
      theta_ref, theta, l, kappa_ref);                    // 笛卡尔坐标系转换为frenet坐标系的成员函数, 计算侧轴l的导数
  const float ddl =                                       // l的二次导数
      CartesianFrenetConverter::CalculateSecondOrderLateralDerivative(
          theta_ref, theta, kappa_ref, kappa, dkappa_ref, l);
  frenet_frame_point->set_dl(dl);                         // 复制给一个frenet坐标系中的一个frame点
  frenet_frame_point->set_ddl(ddl);
  return true;
}

bool DPRoadGraph::IsValidCurve(const QuinticPolynomialCurve1d &curve) const {
  constexpr float kMaxLateralDistance = 20.0;
  for (float s = 0.0; s < curve.ParamLength(); s += 2.0) {
    const float l = curve.Evaluate(0, s);
    if (std::fabs(l) > kMaxLateralDistance) {
      return false;
    }
  }
  return true;
}

void DPRoadGraph::GetCurveCost(TrajectoryCost trajectory_cost,
                               const QuinticPolynomialCurve1d &curve,
                               const float start_s, const float end_s,
                               const uint32_t curr_level,
                               const uint32_t total_level,
                               ComparableCost *cost) {
  *cost =
      trajectory_cost.Calculate(curve, start_s, end_s, curr_level, total_level);
}

bool DPRoadGraph::HasSidepass() {
  const auto &path_decision = reference_line_info_.path_decision();
  for (const auto &obstacle : path_decision.path_obstacles().Items()) {
    if (obstacle->LateralDecision().has_sidepass()) {
      sidepass_ = obstacle->LateralDecision().sidepass();
      return true;
    }
  }
  return false;
}

}  // namespace planning
}  // namespace apollo
