/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "iceberg/parquet/apply_name_mapping_visitor_internal.h"

#include <algorithm>
#include <memory>
#include <ranges>
#include <string_view>

#include <parquet/schema.h>

#include "iceberg/name_mapping.h"
#include "iceberg/result.h"
#include "iceberg/util/checked_cast.h"
#include "iceberg/util/formatter.h"
#include "iceberg/util/macros.h"

namespace iceberg::parquet {

namespace {

constexpr std::string_view kListElementName = "element";
constexpr std::string_view kMapKeyName = "key";
constexpr std::string_view kMapValueName = "value";

}  // namespace

ApplyNameMappingVisitor::ApplyNameMappingVisitor(const NameMapping& name_mapping)
    : name_mapping_(name_mapping) {}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::Apply(
    const ::parquet::schema::NodePtr& original_node, const NameMapping& name_mapping) {
  ApplyNameMappingVisitor visitor(name_mapping);
  return ParquetTypeVisitor<::parquet::schema::NodePtr>::Visit(original_node, visitor);
}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::Message(
    const ::parquet::schema::GroupNode& message,
    std::vector<::parquet::schema::NodePtr>&& fields) {
  const auto filtered_fields = FilterNullNodes(std::move(fields));
  return ::parquet::schema::GroupNode::Make(
      message.name(), message.repetition(), std::move(filtered_fields), 
      nullptr, -1);
}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::Struct(
    const ::parquet::schema::GroupNode& struct_node,
    std::vector<::parquet::schema::NodePtr>&& fields) {
  const auto current_path = CurrentPath();
  const auto mapped_field = FindMappedField(current_path);
  
  // Filter out null nodes but continue processing even if some fields couldn't be mapped
  const auto filtered_fields = FilterNullNodes(std::move(fields));
  auto new_struct = ::parquet::schema::GroupNode::Make(
      struct_node.name(), struct_node.repetition(), std::move(filtered_fields),
      nullptr, -1);
  
  if (mapped_field.has_value() && mapped_field->get().field_id.has_value()) {
    return CreateNodeWithFieldId(new_struct, mapped_field->get().field_id.value());
  }
  
  // Return struct without field ID if mapping not found - this allows partial mapping
  return new_struct;
}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::List(
    const ::parquet::schema::GroupNode& list_node,
    ::parquet::schema::NodePtr&& element) {
  if (!element) {
    return InvalidSchema("List type must have element field");
  }
  
  auto current_path = CurrentPath();
  auto mapped_field = FindMappedField(current_path);
  
  auto list_element_type = DetermineListElementType(list_node);
  ::parquet::schema::NodePtr new_list;
  
  if (list_element_type && list_element_type->repetition() == ::parquet::Repetition::REPEATED) {
    // Two-level list
    std::vector<::parquet::schema::NodePtr> list_fields = {element};
    new_list = ::parquet::schema::GroupNode::Make(
        list_node.name(), list_node.repetition(), std::move(list_fields),
        ::parquet::LogicalType::List(), -1);
  } else {
    // Three-level list
    std::vector<::parquet::schema::NodePtr> element_fields = {element};
    auto repeated_group = ::parquet::schema::GroupNode::Make(
        list_node.field(0)->name(), ::parquet::Repetition::REPEATED,
        std::move(element_fields), nullptr, -1);
    
    std::vector<::parquet::schema::NodePtr> list_fields = {repeated_group};
    new_list = ::parquet::schema::GroupNode::Make(
        list_node.name(), list_node.repetition(), std::move(list_fields),
        ::parquet::LogicalType::List(), -1);
  }
  
  if (mapped_field.has_value() && mapped_field->get().field_id.has_value()) {
    return CreateNodeWithFieldId(new_list, mapped_field->get().field_id.value());
  }
  
  return new_list;
}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::Map(
    const ::parquet::schema::GroupNode& map_node,
    std::optional<::parquet::schema::NodePtr>&& key,
    std::optional<::parquet::schema::NodePtr>&& value) {
  // Allow partial mapping for map fields - at least one of key or value should be present
  if (!key && !value) {
    // Return the original map node without field ID if neither key nor value is mapped
    auto current_path = CurrentPath();
    auto mapped_field = FindMappedField(current_path);
    
    std::vector<::parquet::schema::NodePtr> key_value_fields;  // Empty fields for unmapped map
    auto repeated_key_value = ::parquet::schema::GroupNode::Make(
        map_node.field(0)->name(), ::parquet::Repetition::REPEATED,
        std::move(key_value_fields), nullptr, -1);
    
    std::vector<::parquet::schema::NodePtr> map_fields = {repeated_key_value};
    auto new_map = ::parquet::schema::GroupNode::Make(
        map_node.name(), map_node.repetition(), std::move(map_fields),
        ::parquet::LogicalType::Map(), -1);
    
    if (mapped_field.has_value() && mapped_field->get().field_id.has_value()) {
      return CreateNodeWithFieldId(new_map, mapped_field->get().field_id.value());
    }
    
    return new_map;
  }
  
  auto current_path = CurrentPath();
  auto mapped_field = FindMappedField(current_path);
  
  std::vector<::parquet::schema::NodePtr> key_value_fields;
  if (key) {
    key_value_fields.push_back(std::move(*key));
  }
  if (value) {
    key_value_fields.push_back(std::move(*value));
  }
  
  auto repeated_key_value = ::parquet::schema::GroupNode::Make(
      map_node.field(0)->name(), ::parquet::Repetition::REPEATED,
      std::move(key_value_fields), nullptr, -1);
  
  std::vector<::parquet::schema::NodePtr> map_fields = {repeated_key_value};
  auto new_map = ::parquet::schema::GroupNode::Make(
      map_node.name(), map_node.repetition(), std::move(map_fields),
      ::parquet::LogicalType::Map(), -1);
  
  if (mapped_field.has_value() && mapped_field->get().field_id.has_value()) {
    return CreateNodeWithFieldId(new_map, mapped_field->get().field_id.value());
  }
  
  return new_map;
}

Result<::parquet::schema::NodePtr> ApplyNameMappingVisitor::Primitive(
    const ::parquet::schema::PrimitiveNode& primitive) {
  auto current_path = CurrentPath();
  auto mapped_field = FindMappedField(current_path);
  
  if (mapped_field.has_value() && mapped_field->get().field_id.has_value()) {
    return CreateNodeWithFieldId(
        ::parquet::schema::PrimitiveNode::Make(
            primitive.name(), primitive.repetition(), primitive.logical_type(),
            primitive.physical_type(), primitive.type_length(), -1),
        mapped_field->get().field_id.value());
  }
  
  return ::parquet::schema::PrimitiveNode::Make(
      primitive.name(), primitive.repetition(), primitive.logical_type(),
      primitive.physical_type(), primitive.type_length(), -1);
}

void ApplyNameMappingVisitor::BeforeField(const ::parquet::schema::Node& field) {
  field_names_.push_back(field.name());
}

void ApplyNameMappingVisitor::AfterField(const ::parquet::schema::Node& field) {
  field_names_.pop_back();
}

void ApplyNameMappingVisitor::BeforeElementField(const ::parquet::schema::Node& element) {
  field_names_.emplace_back(kListElementName);
}

void ApplyNameMappingVisitor::AfterElementField(const ::parquet::schema::Node& element) {
  field_names_.pop_back();
}

void ApplyNameMappingVisitor::BeforeKeyField(const ::parquet::schema::Node& key) {
  field_names_.emplace_back(kMapKeyName);
}

void ApplyNameMappingVisitor::AfterKeyField(const ::parquet::schema::Node& key) {
  field_names_.pop_back();
}

void ApplyNameMappingVisitor::BeforeValueField(const ::parquet::schema::Node& value) {
  field_names_.emplace_back(kMapValueName);
}

void ApplyNameMappingVisitor::AfterValueField(const ::parquet::schema::Node& value) {
  field_names_.pop_back();
}

void ApplyNameMappingVisitor::BeforeRepeatedElement(const ::parquet::schema::Node& element) {
  // Do not add the repeated element's name
}

void ApplyNameMappingVisitor::AfterRepeatedElement(const ::parquet::schema::Node& element) {
  // Do not remove the repeated element's name
}

void ApplyNameMappingVisitor::BeforeRepeatedKeyValue(const ::parquet::schema::Node& key_value) {
  // Do not add the repeated element's name
}

void ApplyNameMappingVisitor::AfterRepeatedKeyValue(const ::parquet::schema::Node& key_value) {
  // Do not remove the repeated element's name
}

std::optional<MappedFieldConstRef> ApplyNameMappingVisitor::FindMappedField(
    const std::vector<std::string>& path) const {
  auto field_opt = name_mapping_.Find(path);
  if (!field_opt.has_value()) {
    return std::nullopt;
  }
  
  const MappedField& field = field_opt.value().get();
  if (!field.field_id.has_value()) {
    return std::nullopt;
  }
  
  return field_opt;
}

std::vector<std::string> ApplyNameMappingVisitor::CurrentPath() const {
  return std::vector<std::string>(field_names_.rbegin(), field_names_.rend());
}

::parquet::schema::NodePtr ApplyNameMappingVisitor::CreateNodeWithFieldId(
    const ::parquet::schema::NodePtr& original_node, int32_t field_id) {
  if (original_node->is_group()) {
    const auto& group_node = 
        internal::checked_cast<const ::parquet::schema::GroupNode&>(*original_node);
    
    std::vector<::parquet::schema::NodePtr> fields;
    fields.reserve(group_node.field_count());
    for (int i = 0; i < group_node.field_count(); ++i) {
      fields.push_back(group_node.field(i));
    }
    
    return ::parquet::schema::GroupNode::Make(
        group_node.name(), group_node.repetition(), std::move(fields),
        nullptr, field_id);
  } else {
    const auto& primitive_node = 
        internal::checked_cast<const ::parquet::schema::PrimitiveNode&>(*original_node);
    
    return ::parquet::schema::PrimitiveNode::Make(
        primitive_node.name(), primitive_node.repetition(),
        primitive_node.logical_type(), primitive_node.physical_type(),
        primitive_node.type_length(), field_id);
  }
}

std::vector<::parquet::schema::NodePtr> ApplyNameMappingVisitor::FilterNullNodes(
    std::vector<::parquet::schema::NodePtr>&& nodes) {
  auto filtered = nodes 
    | std::views::filter([](const auto& node) { return node != nullptr; });
  
  return std::vector<::parquet::schema::NodePtr>(filtered.begin(), filtered.end());
}

::parquet::schema::NodePtr ApplyNameMappingVisitor::DetermineListElementType(
    const ::parquet::schema::GroupNode& list_node) {
  if (list_node.field_count() != 1) {
    return nullptr;
  }
  
  const auto& repeated_element = list_node.field(0);
  if (repeated_element->repetition() == ::parquet::Repetition::REPEATED) {
    if (repeated_element->is_group()) {
      const auto& group = 
          internal::checked_cast<const ::parquet::schema::GroupNode&>(*repeated_element);
      if (group.field_count() == 1) {
        return group.field(0);
      }
    }
    return repeated_element;
  }
  
  return repeated_element;
}

}  // namespace iceberg::parquet