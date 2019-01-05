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

#include "modules/planning/common/planning_context.h"

#include "modules/common/adapters/adapter_manager.h"
#include "modules/planning/common/planning_gflags.h"

namespace apollo {
namespace planning {

using common::adapter::AdapterManager;

PlanningContext::PlanningContext() {}

void DumpPlanningContext() {    // 切换到planning的上下文中
  AdapterManager::GetLocalization()->DumpLatestMessage();      // 获取定位的信息
  AdapterManager::GetChassis()->DumpLatestMessage();           // 获取底盘的信息
  AdapterManager::GetRoutingResponse()->DumpLatestMessage();   // 获取ronting的信息
  AdapterManager::GetPrediction()->DumpLatestMessage();        // 获取预测的信息
}

void PlanningContext::Clear() { planning_status_.Clear(); }    // 所有的状态都清零

}  // namespace planning
}  // namespace apollo
