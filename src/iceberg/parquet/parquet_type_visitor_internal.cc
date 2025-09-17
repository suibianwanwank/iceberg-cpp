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

#include "iceberg/parquet/parquet_type_visitor_internal.h"

#include <algorithm>
#include <iterator>
#include <string_view>

#include <parquet/schema.h>

#include "iceberg/result.h"
#include "iceberg/util/checked_cast.h"
#include "iceberg/util/macros.h"

namespace iceberg::parquet {

namespace {

constexpr std::string_view kListElementName = "element";
constexpr std::string_view kMapKeyName = "key";
constexpr std::string_view kMapValueName = "value";

}  // namespace

template <typename T>
Result<T> ParquetTypeVisitor<T>::Visit(const ::parquet::schema::NodePtr& node,
                                       ParquetTypeVisitor<T>& visitor) {
  return VisitNode(node, visitor);
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeField(const ::parquet::schema::Node& field) {
  field_names_.push_back(field.name());
}

template <typename T>
void ParquetTypeVisitor<T>::AfterField(const ::parquet::schema::Node& field) {
  field_names_.pop_back();
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeRepeatedElement(const ::parquet::schema::Node& element) {
  BeforeField(element);
}

template <typename T>
void ParquetTypeVisitor<T>::AfterRepeatedElement(const ::parquet::schema::Node& element) {
  AfterField(element);
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeElementField(const ::parquet::schema::Node& element) {
  field_names_.emplace_back(kListElementName);
}

template <typename T>
void ParquetTypeVisitor<T>::AfterElementField(const ::parquet::schema::Node& element) {
  field_names_.pop_back();
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeRepeatedKeyValue(const ::parquet::schema::Node& key_value) {
  // Do not add the repeated element's name
}

template <typename T>
void ParquetTypeVisitor<T>::AfterRepeatedKeyValue(const ::parquet::schema::Node& key_value) {
  // Do not remove the repeated element's name
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeKeyField(const ::parquet::schema::Node& key) {
  field_names_.emplace_back(kMapKeyName);
}

template <typename T>
void ParquetTypeVisitor<T>::AfterKeyField(const ::parquet::schema::Node& key) {
  field_names_.pop_back();
}

template <typename T>
void ParquetTypeVisitor<T>::BeforeValueField(const ::parquet::schema::Node& value) {
  field_names_.emplace_back(kMapValueName);
}

template <typename T>
void ParquetTypeVisitor<T>::AfterValueField(const ::parquet::schema::Node& value) {
  field_names_.pop_back();
}

template <typename T>
std::vector<std::string> ParquetTypeVisitor<T>::CurrentPath() const {
  return std::vector<std::string>(field_names_.rbegin(), field_names_.rend());
}

template <typename T>
std::vector<std::string> ParquetTypeVisitor<T>::Path(const std::string& name) const {
  auto path = CurrentPath();
  path.push_back(name);
  return path;
}

template <typename T>
Result<T> ParquetTypeVisitor<T>::VisitNode(const ::parquet::schema::NodePtr& node,
                                           ParquetTypeVisitor<T>& visitor) {
  if (node->is_primitive()) {
    const auto& primitive = 
        internal::checked_cast<const ::parquet::schema::PrimitiveNode&>(*node);
    return visitor.Primitive(primitive);
  }

  const auto& group = internal::checked_cast<const ::parquet::schema::GroupNode&>(*node);
  
  if (IsListType(group)) {
    return VisitList(group, visitor);
  }
  
  if (IsMapType(group)) {
    return VisitMap(group, visitor);
  }

  // Handle message type (root schema)
  if (group.name() == "schema" || !group.is_repeated()) {
    ICEBERG_ASSIGN_OR_RAISE(auto fields, VisitFields(group, visitor));
    return visitor.Message(group, std::move(fields));
  }
  
  // Regular struct type
  ICEBERG_ASSIGN_OR_RAISE(auto fields, VisitFields(group, visitor));
  return visitor.Struct(group, std::move(fields));
}

template <typename T>
Result<std::vector<T>> ParquetTypeVisitor<T>::VisitFields(
    const ::parquet::schema::GroupNode& group, ParquetTypeVisitor<T>& visitor) {
  std::vector<T> results;
  results.reserve(group.field_count());
  
  for (int i = 0; i < group.field_count(); ++i) {
    const auto& field = group.field(i);
    visitor.BeforeField(*field);
    
    ICEBERG_ASSIGN_OR_RAISE(auto result, VisitNode(field, visitor));
    results.emplace_back(std::move(result));
    
    visitor.AfterField(*field);
  }
  
  return results;
}

template <typename T>
Result<T> ParquetTypeVisitor<T>::VisitList(const ::parquet::schema::GroupNode& list_node,
                                           ParquetTypeVisitor<T>& visitor) {
  if (list_node.field_count() != 1) {
    return InvalidSchema("Invalid list: does not contain single repeated field: {}",
                         list_node.name());
  }
  
  const auto& repeated_element = list_node.field(0);
  if (repeated_element->repetition() != ::parquet::Repetition::REPEATED) {
    return InvalidSchema("Invalid list: inner group is not repeated");
  }
  
  auto list_element = DetermineListElementType(list_node);
  if (list_element->repetition() == ::parquet::Repetition::REPEATED) {
    ICEBERG_ASSIGN_OR_RAISE(auto element_result, VisitListElement(list_element, visitor));
    return visitor.List(list_node, std::move(element_result));
  } else {
    return VisitThreeLevelList(list_node, *repeated_element, *list_element, visitor);
  }
}

template <typename T>
Result<T> ParquetTypeVisitor<T>::VisitThreeLevelList(
    const ::parquet::schema::GroupNode& list_node,
    const ::parquet::schema::Node& repeated,
    const ::parquet::schema::Node& list_element,
    ParquetTypeVisitor<T>& visitor) {
  visitor.BeforeRepeatedElement(repeated);
  
  // Create a copy of the list element node as a shared_ptr
  ::parquet::schema::NodePtr list_element_ptr;
  if (list_element.is_primitive()) {
    const auto& primitive = 
        internal::checked_cast<const ::parquet::schema::PrimitiveNode&>(list_element);
    list_element_ptr = ::parquet::schema::PrimitiveNode::Make(
        primitive.name(), primitive.repetition(), primitive.logical_type(),
        primitive.physical_type(), primitive.type_length(), primitive.field_id());
  } else {
    const auto& group = 
        internal::checked_cast<const ::parquet::schema::GroupNode&>(list_element);
    std::vector<::parquet::schema::NodePtr> fields;
    for (int i = 0; i < group.field_count(); ++i) {
      fields.push_back(group.field(i));
    }
    list_element_ptr = ::parquet::schema::GroupNode::Make(
        group.name(), group.repetition(), std::move(fields), 
        group.logical_type(), group.field_id());
  }
  
  ICEBERG_ASSIGN_OR_RAISE(auto element_result, VisitListElement(list_element_ptr, visitor));
  auto result = visitor.List(list_node, std::move(element_result));
  
  visitor.AfterRepeatedElement(repeated);
  return result;
}

template <typename T>
Result<T> ParquetTypeVisitor<T>::VisitListElement(
    const ::parquet::schema::NodePtr& list_element, ParquetTypeVisitor<T>& visitor) {
  visitor.BeforeElementField(*list_element);
  auto result = VisitNode(list_element, visitor);
  visitor.AfterElementField(*list_element);
  return result;
}

template <typename T>
Result<T> ParquetTypeVisitor<T>::VisitMap(const ::parquet::schema::GroupNode& map_node,
                                          ParquetTypeVisitor<T>& visitor) {
  if (map_node.repetition() == ::parquet::Repetition::REPEATED) {
    return InvalidSchema("Invalid map: top-level group is repeated: {}", map_node.name());
  }
  
  if (map_node.field_count() != 1) {
    return InvalidSchema("Invalid map: does not contain single repeated field: {}",
                         map_node.name());
  }
  
  const auto& repeated_key_value_node = map_node.field(0);
  if (repeated_key_value_node->repetition() != ::parquet::Repetition::REPEATED) {
    return InvalidSchema("Invalid map: inner group is not repeated");
  }
  
  if (!repeated_key_value_node->is_group()) {
    return InvalidSchema("Invalid map: repeated field is not a group");
  }
  
  const auto& repeated_key_value = 
      internal::checked_cast<const ::parquet::schema::GroupNode&>(*repeated_key_value_node);
  
  if (repeated_key_value.field_count() > 2) {
    return InvalidSchema("Invalid map: repeated group has more than 2 fields");
  }
  
  visitor.BeforeRepeatedKeyValue(repeated_key_value);
  
  std::optional<T> key_result;
  std::optional<T> value_result;
  
  switch (repeated_key_value.field_count()) {
    case 2: {
      // Both key and value are projected
      const auto& key_type = repeated_key_value.field(0);
      visitor.BeforeKeyField(*key_type);
      ICEBERG_ASSIGN_OR_RAISE(auto key_res, VisitNode(key_type, visitor));
      key_result = std::move(key_res);
      visitor.AfterKeyField(*key_type);
      
      const auto& value_type = repeated_key_value.field(1);
      visitor.BeforeValueField(*value_type);
      ICEBERG_ASSIGN_OR_RAISE(auto value_res, VisitNode(value_type, visitor));
      value_result = std::move(value_res);
      visitor.AfterValueField(*value_type);
      break;
    }
    case 1: {
      // Only key or value is projected, determine by name
      const auto& key_or_value = repeated_key_value.field(0);
      if (key_or_value->name() == kMapKeyName) {
        visitor.BeforeKeyField(*key_or_value);
        ICEBERG_ASSIGN_OR_RAISE(auto key_res, VisitNode(key_or_value, visitor));
        key_result = std::move(key_res);
        visitor.AfterKeyField(*key_or_value);
      } else {
        visitor.BeforeValueField(*key_or_value);
        ICEBERG_ASSIGN_OR_RAISE(auto value_res, VisitNode(key_or_value, visitor));
        value_result = std::move(value_res);
        visitor.AfterValueField(*key_or_value);
      }
      break;
    }
    default:
      // Both results remain nullopt
      break;
  }
  
  auto result = visitor.Map(map_node, std::move(key_result), std::move(value_result));
  visitor.AfterRepeatedKeyValue(repeated_key_value);
  return result;
}

template <typename T>
::parquet::schema::NodePtr ParquetTypeVisitor<T>::DetermineListElementType(
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

template <typename T>
bool ParquetTypeVisitor<T>::IsListType(const ::parquet::schema::GroupNode& group) {
  return group.logical_type() && 
         group.logical_type()->type() == ::parquet::LogicalType::Type::LIST;
}

template <typename T>
bool ParquetTypeVisitor<T>::IsMapType(const ::parquet::schema::GroupNode& group) {
  return group.logical_type() && 
         group.logical_type()->type() == ::parquet::LogicalType::Type::MAP;
}

// Explicit instantiation for the types we need
template class ParquetTypeVisitor<::parquet::schema::NodePtr>;

}  // namespace iceberg::parquet