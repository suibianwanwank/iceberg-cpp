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

#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <parquet/schema.h>

#include "iceberg/result.h"

namespace iceberg::parquet {

/// \brief Base visitor class for traversing Parquet schema nodes.
///
/// This class provides the framework for implementing schema transformations
/// using the Visitor pattern, similar to Java's ParquetTypeVisitor.
template <typename T>
class ParquetTypeVisitor {
 public:
  virtual ~ParquetTypeVisitor() = default;

  /// \brief Visit a Parquet schema node and return the transformation result.
  ///
  /// \param node The Parquet schema node to visit.
  /// \return The transformation result or an error.
  [[nodiscard]] static Result<T> Visit(const ::parquet::schema::NodePtr& node,
                                       ParquetTypeVisitor<T>& visitor);

  /// \brief Visit a message (root schema) node.
  ///
  /// \param message The message node.
  /// \param fields The transformed child fields.
  /// \return The transformation result.
  [[nodiscard]] virtual Result<T> Message(const ::parquet::schema::GroupNode& message,
                                           std::vector<T>&& fields) = 0;

  /// \brief Visit a struct (group) node.
  ///
  /// \param struct_node The struct node.
  /// \param fields The transformed child fields.
  /// \return The transformation result.
  [[nodiscard]] virtual Result<T> Struct(const ::parquet::schema::GroupNode& struct_node,
                                          std::vector<T>&& fields) = 0;

  /// \brief Visit a list node.
  ///
  /// \param list_node The list node.
  /// \param element The transformed element type.
  /// \return The transformation result.
  [[nodiscard]] virtual Result<T> List(const ::parquet::schema::GroupNode& list_node,
                                        T&& element) = 0;

  /// \brief Visit a map node.
  ///
  /// \param map_node The map node.
  /// \param key The transformed key type (may be nullopt if not projected).
  /// \param value The transformed value type (may be nullopt if not projected).
  /// \return The transformation result.
  [[nodiscard]] virtual Result<T> Map(const ::parquet::schema::GroupNode& map_node,
                                       std::optional<T>&& key,
                                       std::optional<T>&& value) = 0;

  /// \brief Visit a primitive node.
  ///
  /// \param primitive The primitive node.
  /// \return The transformation result.
  [[nodiscard]] virtual Result<T> Primitive(const ::parquet::schema::PrimitiveNode& primitive) = 0;

  /// \brief Called before visiting a field.
  ///
  /// \param field The field about to be visited.
  virtual void BeforeField(const ::parquet::schema::Node& field);

  /// \brief Called after visiting a field.
  ///
  /// \param field The field that was visited.
  virtual void AfterField(const ::parquet::schema::Node& field);

  /// \brief Called before visiting a repeated element in a list.
  ///
  /// \param element The repeated element.
  virtual void BeforeRepeatedElement(const ::parquet::schema::Node& element);

  /// \brief Called after visiting a repeated element in a list.
  ///
  /// \param element The repeated element.
  virtual void AfterRepeatedElement(const ::parquet::schema::Node& element);

  /// \brief Called before visiting a list element field.
  ///
  /// \param element The list element field.
  virtual void BeforeElementField(const ::parquet::schema::Node& element);

  /// \brief Called after visiting a list element field.
  ///
  /// \param element The list element field.
  virtual void AfterElementField(const ::parquet::schema::Node& element);

  /// \brief Called before visiting a repeated key-value pair in a map.
  ///
  /// \param key_value The repeated key-value pair.
  virtual void BeforeRepeatedKeyValue(const ::parquet::schema::Node& key_value);

  /// \brief Called after visiting a repeated key-value pair in a map.
  ///
  /// \param key_value The repeated key-value pair.
  virtual void AfterRepeatedKeyValue(const ::parquet::schema::Node& key_value);

  /// \brief Called before visiting a map key field.
  ///
  /// \param key The map key field.
  virtual void BeforeKeyField(const ::parquet::schema::Node& key);

  /// \brief Called after visiting a map key field.
  ///
  /// \param key The map key field.
  virtual void AfterKeyField(const ::parquet::schema::Node& key);

  /// \brief Called before visiting a map value field.
  ///
  /// \param value The map value field.
  virtual void BeforeValueField(const ::parquet::schema::Node& value);

  /// \brief Called after visiting a map value field.
  ///
  /// \param value The map value field.
  virtual void AfterValueField(const ::parquet::schema::Node& value);

 protected:
  /// \brief Get the current field path.
  ///
  /// \return The current field path as a vector of strings.
  [[nodiscard]] std::vector<std::string> CurrentPath() const;

  /// \brief Get the field path with an additional name appended.
  ///
  /// \param name The name to append.
  /// \return The field path with the name appended.
  [[nodiscard]] std::vector<std::string> Path(const std::string& name) const;

 private:
  /// \brief Visitor implementation for different node types.
  static Result<T> VisitNode(const ::parquet::schema::NodePtr& node,
                             ParquetTypeVisitor<T>& visitor);

  /// \brief Visit fields of a group node.
  static Result<std::vector<T>> VisitFields(const ::parquet::schema::GroupNode& group,
                                            ParquetTypeVisitor<T>& visitor);

  /// \brief Visit a list node implementation.
  static Result<T> VisitList(const ::parquet::schema::GroupNode& list_node,
                             ParquetTypeVisitor<T>& visitor);

  /// \brief Visit a three-level list node implementation.
  static Result<T> VisitThreeLevelList(const ::parquet::schema::GroupNode& list_node,
                                       const ::parquet::schema::Node& repeated,
                                       const ::parquet::schema::Node& list_element,
                                       ParquetTypeVisitor<T>& visitor);

  /// \brief Visit a list element.
  static Result<T> VisitListElement(const ::parquet::schema::NodePtr& list_element,
                                    ParquetTypeVisitor<T>& visitor);

  /// \brief Visit a map node implementation.
  static Result<T> VisitMap(const ::parquet::schema::GroupNode& map_node,
                            ParquetTypeVisitor<T>& visitor);

  /// \brief Determine the list element type.
  static ::parquet::schema::NodePtr DetermineListElementType(
      const ::parquet::schema::GroupNode& list_node);

  /// \brief Check if a node is a list type.
  static bool IsListType(const ::parquet::schema::GroupNode& group);

  /// \brief Check if a node is a map type.
  static bool IsMapType(const ::parquet::schema::GroupNode& group);

  /// \brief Stack to track field names for path construction.
  std::deque<std::string> field_names_;
};

}  // namespace iceberg::parquet