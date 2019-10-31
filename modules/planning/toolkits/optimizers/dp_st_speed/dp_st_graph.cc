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
 * @file dp_st_graph.cc
 **/

#include "modules/planning/toolkits/optimizers/dp_st_speed/dp_st_graph.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "modules/common/log.h"
#include "modules/common/math/vec2d.h"
#include "modules/common/proto/pnc_point.pb.h"
#include "modules/common/util/thread_pool.h"

#include "modules/planning/common/planning_gflags.h"

namespace apollo {
namespace planning {

using apollo::common::ErrorCode;
using apollo::common::SpeedPoint;
using apollo::common::Status;
using apollo::common::VehicleParam;
using apollo::common::math::Vec2d;
using apollo::common::util::ThreadPool;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

bool CheckOverlapOnDpStGraph(const std::vector<const StBoundary*>& boundaries,
                             const StGraphPoint& p1, const StGraphPoint& p2) {
  const common::math::LineSegment2d seg(p1.point(), p2.point());
  for (const auto* boundary : boundaries) {
    if (boundary->boundary_type() == StBoundary::BoundaryType::KEEP_CLEAR) {
      continue;
    }
    if (boundary->HasOverlap(seg)) {
      return true;
    }
  }
  return false;
}
}  // namespace

DpStGraph::DpStGraph(const StGraphData& st_graph_data,
                     const DpStSpeedConfig& dp_config,
                     const std::vector<const PathObstacle*>& obstacles,
                     const common::TrajectoryPoint& init_point,
                     const SLBoundary& adc_sl_boundary)
    : st_graph_data_(st_graph_data),
      dp_st_speed_config_(dp_config),
      obstacles_(obstacles),
      init_point_(init_point),
      dp_st_cost_(dp_config, obstacles, init_point_),
      adc_sl_boundary_(adc_sl_boundary) {
  dp_st_speed_config_.set_total_path_length(
      std::fmin(dp_st_speed_config_.total_path_length(),
                st_graph_data_.path_data_length()));
  unit_s_ = dp_st_speed_config_.total_path_length() /
            (dp_st_speed_config_.matrix_dimension_s() - 1);
  unit_t_ = dp_st_speed_config_.total_time() /
            (dp_st_speed_config_.matrix_dimension_t() - 1);
}

Status DpStGraph::Search(SpeedData* const speed_data) {
  //每个low_s,high_s以及他们生成的stboundary是从0开始的。也就是说 后续的st图search的原点坐标为(0,0)，详细见st_boundary_mapper.cc，搜索path_s
  constexpr float kBounadryEpsilon = 1e-2;
  for (const auto& boundary : st_graph_data_.st_boundaries()) {
    if (boundary->boundary_type() == StBoundary::BoundaryType::KEEP_CLEAR) {
      continue;
    }
    //检验初始状态（如起点是否已经撞到了障碍物） 若状态异常则目标位移设为0制动
    if (boundary->IsPointInBoundary({0.0, 0.0}) ||
        (std::fabs(boundary->min_t()) < kBounadryEpsilon &&
         std::fabs(boundary->min_s()) < kBounadryEpsilon)) {
      std::vector<SpeedPoint> speed_profile;
      float t = 0.0;
      for (int i = 0; i < dp_st_speed_config_.matrix_dimension_t();
           ++i, t += unit_t_) {
        SpeedPoint speed_point;//成员包括 s t v a da
        speed_point.set_s(0.0);//状态异常，进行制动
        speed_point.set_t(t);
        speed_profile.emplace_back(speed_point);
      }
      speed_data->set_speed_vector(speed_profile);
      return Status::OK();
    }
  }

  if (st_graph_data_.st_boundaries().empty()) {//ST图中无任何障碍物，输出默认速度
    ADEBUG << "No path obstacles, dp_st_graph output default speed profile.";
    std::vector<SpeedPoint> speed_profile;
    float s = 0.0;
    float t = 0.0;
    for (int i = 0; i < dp_st_speed_config_.matrix_dimension_t() &&
                    i < dp_st_speed_config_.matrix_dimension_s();
         ++i, t += unit_t_, s += unit_s_) {
      SpeedPoint speed_point;
      speed_point.set_s(s);
      speed_point.set_t(t);
      const float v_default = unit_s_ / unit_t_;
      speed_point.set_v(v_default);
      speed_point.set_a(0.0);
      speed_profile.emplace_back(std::move(speed_point));
    }
    speed_data->set_speed_vector(std::move(speed_profile));
    return Status::OK();
  }

  if (!InitCostTable().ok()) {
    const std::string msg = "Initialize cost table failed.";
    AERROR << msg;
    return Status(ErrorCode::PLANNING_ERROR, msg);
  }

  if (!CalculateTotalCost().ok()) {
    const std::string msg = "Calculate total cost failed.";
    AERROR << msg;
    return Status(ErrorCode::PLANNING_ERROR, msg);
  }

  if (!RetrieveSpeedProfile(speed_data).ok()) {
    const std::string msg = "Retrieve best speed profile failed.";
    AERROR << msg;
    return Status(ErrorCode::PLANNING_ERROR, msg);
  }
  return Status::OK();
}

Status DpStGraph::InitCostTable() {
  uint32_t dim_s = dp_st_speed_config_.matrix_dimension_s();//150m
  uint32_t dim_t = dp_st_speed_config_.matrix_dimension_t();//8s
  DCHECK_GT(dim_s, 2);
  DCHECK_GT(dim_t, 2);
  cost_table_ = std::vector<std::vector<StGraphPoint>>(
      dim_t, std::vector<StGraphPoint>(dim_s, StGraphPoint()));
//构造st图，dimt行，dims列，每个元素都是StgraphPoint，然后两个for循环对其初始化
  float curr_t = 0.0;
  for (uint32_t i = 0; i < cost_table_.size(); ++i, curr_t += unit_t_) {
    auto& cost_table_i = cost_table_[i];
    float curr_s = 0.0;
    for (uint32_t j = 0; j < cost_table_i.size(); ++j, curr_s += unit_s_) {
      cost_table_i[j].Init(i, j, STPoint(curr_s, curr_t));//初始化
    }
  }
  return Status::OK();
}

Status DpStGraph::CalculateTotalCost() {
  // col and row are for STGraph
  // t corresponding to col
  // s corresponding to row
  uint32_t next_highest_row = 0;
  uint32_t next_lowest_row = 0;
  //c 代表st图中的t，r代表st图中的s 最外层的for循环用c遍历t，内层for循环用r遍历s。
  //c和r实际是ST图考虑resolution后的坐标，实际s=c*unit_s 实际t=r*unit_t
  for (size_t c = 0; c < cost_table_.size(); ++c) {
    int highest_row = 0;
    int lowest_row = cost_table_.back().size() - 1;

    int count = next_highest_row - next_lowest_row + 1;
    if (count > 0) {
      std::vector<std::future<void>> futures;

      for (uint32_t r = next_lowest_row; r <= next_highest_row; ++r) {
        if (FLAGS_enable_multi_thread_in_dp_st_graph) {
          futures.push_back(ThreadPool::pool()->push(
              std::bind(&DpStGraph::CalculateCostAt, this, c, r)));
        } else {
          CalculateCostAt(c, r);//根据dp算法，找到坐标为(t=c,s=r)的点的父节点，并计算到达（c,r）的最小cost。
        }
      }

      for (const auto& f : futures) {
        f.wait();
      }
    }

    for (uint32_t r = next_lowest_row; r <= next_highest_row; ++r) {
      const auto& cost_cr = cost_table_[c][r];
      if (cost_cr.total_cost() < std::numeric_limits<float>::infinity()) {
        int h_r = 0;
        int l_r = 0;
        GetRowRange(cost_cr, &h_r, &l_r);//用于下一循环（c=c+1）确定遍历s的范围 
        highest_row = std::max(highest_row, h_r);//这个变量应该叫next_highest_row（相应的，next_highest_row改为highest_row）比较好理解
        lowest_row = std::min(lowest_row, l_r);
      }
    }
    next_highest_row = highest_row;
    next_lowest_row = lowest_row;
  }

  return Status::OK();
}

void DpStGraph::GetRowRange(const StGraphPoint& point, int* next_highest_row,
                            int* next_lowest_row) { //根据已经确定完cost和父节点的某个point，根据加速度极限和point的速度确定下一循环遍历的s的范围
  float v0 = 0.0;
  if (!point.pre_point()) {
    v0 = init_point_.v();
  } else {
    v0 = (point.index_s() - point.pre_point()->index_s()) * unit_s_ / unit_t_;
  }//计算当前节点速度

  const int max_s_size = cost_table_.back().size() - 1;

  const float speed_coeff = unit_t_ * unit_t_;

  const float delta_s_upper_bound =
      v0 * unit_t_ + vehicle_param_.max_acceleration() * speed_coeff;
  *next_highest_row =
      point.index_s() + static_cast<int>(delta_s_upper_bound / unit_s_);
  if (*next_highest_row >= max_s_size) {
    *next_highest_row = max_s_size;
  }

  const float delta_s_lower_bound = std::fmax(
      0.0, v0 * unit_t_ + vehicle_param_.max_deceleration() * speed_coeff);
  *next_lowest_row =
      point.index_s() + static_cast<int>(delta_s_lower_bound / unit_s_);
  if (*next_lowest_row > max_s_size) {
    *next_lowest_row = max_s_size;
  } else if (*next_lowest_row < 0) {
    *next_lowest_row = 0;
  }
}

void DpStGraph::CalculateCostAt(const uint32_t c, const uint32_t r) {
  //本函数根据dp算法，找到坐标为(t=c,s=r)的点的父节点，并计算到达（c,r）的最小cost。

  //函数 1.先计算t s坐标为[c][r]的节点的obstacle cost，   2.分成c=0，c=1,c=2 用dp算法遍历父节点的cost+edgecost，取最小的为父节点。之所以分类是因为在计算edgecost时 需要计算加速度和jerk，而这些量需要用到前一到两列的节点。
  auto& cost_cr = cost_table_[c][r];//cost_cr为坐标为c(t)，r(s)的StGraphPoint
  //1.先计算当前点的obstacle_cost
  cost_cr.SetObstacleCost(dp_st_cost_.GetObstacleCost(cost_cr));
  if (cost_cr.obstacle_cost() > std::numeric_limits<float>::max()) {
    return;
  }

  const auto& cost_init = cost_table_[0][0];
  //2.分类讨论，计算各个备选父节点到达该节点的总cost，并取最小者为父节点
  if (c == 0) {
    DCHECK_EQ(r, 0) << "Incorrect. Row should be 0 with col = 0. row: " << r;
    cost_cr.SetTotalCost(0.0);
    return;
  }

  float speed_limit =
      st_graph_data_.speed_limit().GetSpeedLimitByS(unit_s_ * r);
  if (c == 1) {
    //t=1时，加速度限值检测，用st图中的s/t求到达当前节点平均速度v1，再减起始节点速度v0。(v1-v0)/t
    const float acc = (r * unit_s_ / unit_t_ - init_point_.v()) / unit_t_;
    if (acc < dp_st_speed_config_.max_deceleration() ||
        acc > dp_st_speed_config_.max_acceleration()) {
      return;
    }

    if (CheckOverlapOnDpStGraph(st_graph_data_.st_boundaries(), cost_cr,
                                cost_init)) {
      return;//检验连线和stboundaries的相交情况
    }
    cost_cr.SetTotalCost(cost_cr.obstacle_cost() + cost_init.total_cost() +
                         CalculateEdgeCostForSecondCol(r, speed_limit));//EdgeCost 代表的是前一节点到当前节点的jerkcost，加速度cost等
    cost_cr.SetPrePoint(cost_init);
    return;
  }

  constexpr float kSpeedRangeBuffer = 0.20;
  const uint32_t max_s_diff =
      static_cast<uint32_t>(FLAGS_planning_upper_speed_limit *
                            (1 + kSpeedRangeBuffer) * unit_t_ / unit_s_);
  const uint32_t r_low = (max_s_diff < r ? r - max_s_diff : 0);

  const auto& pre_col = cost_table_[c - 1];
//计算cost[c][r]时，用dp算法遍历其可能的父节点们，这些父节点在pre_col中（t一定是c-1时刻），s遍历范围为r_low至r。
  if (c == 2) {
    for (uint32_t r_pre = r_low; r_pre <= r; ++r_pre) {//s遍历范围为r_low至r。
      const float acc =
          (r * unit_s_ - 2 * r_pre * unit_s_) / (unit_t_ * unit_t_);
      if (acc < dp_st_speed_config_.max_deceleration() ||
          acc > dp_st_speed_config_.max_acceleration()) {
        continue;
      }

      if (CheckOverlapOnDpStGraph(st_graph_data_.st_boundaries(), cost_cr,
                                  pre_col[r_pre])) {
        continue;
      }

      const float cost = cost_cr.obstacle_cost() + pre_col[r_pre].total_cost() +
                         CalculateEdgeCostForThirdCol(r, r_pre, speed_limit);

      if (cost < cost_cr.total_cost()) {//dp算法，取cost最小的为父节点
        cost_cr.SetTotalCost(cost);
        cost_cr.SetPrePoint(pre_col[r_pre]);//dp算法，取cost最小的为父节点
      }
    }
    return;
  }
  for (uint32_t r_pre = r_low; r_pre <= r; ++r_pre) {//s遍历范围为r_low至r。
    if (std::isinf(pre_col[r_pre].total_cost()) ||
        pre_col[r_pre].pre_point() == nullptr) {
      continue;
    }
    //加速度检测
    const float curr_a = (cost_cr.index_s() * unit_s_ +
                          pre_col[r_pre].pre_point()->index_s() * unit_s_ -
                          2 * pre_col[r_pre].index_s() * unit_s_) /
                         (unit_t_ * unit_t_);
    if (curr_a > vehicle_param_.max_acceleration() ||
        curr_a < vehicle_param_.max_deceleration()) {
      continue;
    }
    //本节点和父节点连线 与 stboundaries的相交情况
    if (CheckOverlapOnDpStGraph(st_graph_data_.st_boundaries(), cost_cr,
                                pre_col[r_pre])) {
      continue;
    }

    uint32_t r_prepre = pre_col[r_pre].pre_point()->index_s();
    const StGraphPoint& prepre_graph_point = cost_table_[c - 2][r_prepre];//父节点的父节点
    if (std::isinf(prepre_graph_point.total_cost())) {
      continue;
    }

    if (!prepre_graph_point.pre_point()) {
      continue;
    }
    const STPoint& triple_pre_point = prepre_graph_point.pre_point()->point();//父节点的父节点的父节点   用于计算EdgeCost （包括jerkcost，加速度cost等）
    const STPoint& prepre_point = prepre_graph_point.point();
    const STPoint& pre_point = pre_col[r_pre].point();
    const STPoint& curr_point = cost_cr.point();
    float cost = cost_cr.obstacle_cost() + pre_col[r_pre].total_cost() +
                 CalculateEdgeCost(triple_pre_point, prepre_point, pre_point,
                                   curr_point, speed_limit);

    if (cost < cost_cr.total_cost()) {
      cost_cr.SetTotalCost(cost);
      cost_cr.SetPrePoint(pre_col[r_pre]);
    }
  }
}

Status DpStGraph::RetrieveSpeedProfile(SpeedData* const speed_data) {
  float min_cost = std::numeric_limits<float>::infinity();
  const StGraphPoint* best_end_point = nullptr;
  for (const StGraphPoint& cur_point : cost_table_.back()) {
    if (!std::isinf(cur_point.total_cost()) &&
        cur_point.total_cost() < min_cost) {
      best_end_point = &cur_point;
      min_cost = cur_point.total_cost();
    }
  }

  for (const auto& row : cost_table_) {
    const StGraphPoint& cur_point = row.back();
    if (!std::isinf(cur_point.total_cost()) &&
        cur_point.total_cost() < min_cost) {
      best_end_point = &cur_point;
      min_cost = cur_point.total_cost();
    }
  }

  if (best_end_point == nullptr) {
    const std::string msg = "Fail to find the best feasible trajectory.";
    AERROR << msg;
    return Status(ErrorCode::PLANNING_ERROR, msg);
  }

  std::vector<SpeedPoint> speed_profile;
  const StGraphPoint* cur_point = best_end_point;
  while (cur_point != nullptr) {
    SpeedPoint speed_point;
    speed_point.set_s(cur_point->point().s());
    speed_point.set_t(cur_point->point().t());
    speed_profile.emplace_back(speed_point);
    cur_point = cur_point->pre_point();
  }
  std::reverse(speed_profile.begin(), speed_profile.end());

  constexpr float kEpsilon = std::numeric_limits<float>::epsilon();
  if (speed_profile.front().t() > kEpsilon ||
      speed_profile.front().s() > kEpsilon) {
    const std::string msg = "Fail to retrieve speed profile.";
    AERROR << msg;
    return Status(ErrorCode::PLANNING_ERROR, msg);
  }
  speed_data->set_speed_vector(speed_profile);
  return Status::OK();
}

float DpStGraph::CalculateEdgeCost(const STPoint& first, const STPoint& second,
                                   const STPoint& third, const STPoint& forth,
                                   const float speed_limit) {
  return dp_st_cost_.GetSpeedCost(third, forth, speed_limit) +
         dp_st_cost_.GetAccelCostByThreePoints(second, third, forth) +
         dp_st_cost_.GetJerkCostByFourPoints(first, second, third, forth);
}

float DpStGraph::CalculateEdgeCostForSecondCol(const uint32_t row,
                                               const float speed_limit) {
  float init_speed = init_point_.v();
  float init_acc = init_point_.a();
  const STPoint& pre_point = cost_table_[0][0].point();
  const STPoint& curr_point = cost_table_[1][row].point();
  return dp_st_cost_.GetSpeedCost(pre_point, curr_point, speed_limit) +
         dp_st_cost_.GetAccelCostByTwoPoints(init_speed, pre_point,
                                             curr_point) +
         dp_st_cost_.GetJerkCostByTwoPoints(init_speed, init_acc, pre_point,
                                            curr_point);
}

float DpStGraph::CalculateEdgeCostForThirdCol(const uint32_t curr_row,
                                              const uint32_t pre_row,
                                              const float speed_limit) {
  float init_speed = init_point_.v();
  const STPoint& first = cost_table_[0][0].point();
  const STPoint& second = cost_table_[1][pre_row].point();
  const STPoint& third = cost_table_[2][curr_row].point();
  return dp_st_cost_.GetSpeedCost(second, third, speed_limit) +
         dp_st_cost_.GetAccelCostByThreePoints(first, second, third) +
         dp_st_cost_.GetJerkCostByThreePoints(init_speed, first, second, third);
}

}  // namespace planning
}  // namespace apollo
