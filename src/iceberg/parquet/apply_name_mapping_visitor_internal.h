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
#include <vector>

#include <parquet/schema.h>

#include "iceberg/name_mapping.h"
#include "iceberg/parquet/parquet_type_visitor_internal.h"
#include "iceberg/result.h"

namespace iceberg::parquet {

/// \brief Visitor that applies name mapping to create a Parquet schema with field IDs.
///
/// This visitor traverses the original Parquet schema and creates a new schema
/// with field IDs applied based on the provided name mapping. It handles
/// complex nested types like lists and maps correctly.
class ApplyNameMappingVisitor : public ParquetTypeVisitor<::parquet::schema::NodePtr> {
 public:
  /// \brief Constructor.
  ///
  /// \param name_mapping The name mapping to apply.
  explicit ApplyNameMappingVisitor(const NameMapping& name_mapping);

  /// \brief Apply name mapping to a Parquet schema node.
  ///
  /// \param original_node The original Parquet schema node.
  /// \param name_mapping The name mapping to apply.
  /// \return A new Parquet schema node with field IDs applied.
  [[nodiscard]] static Result<::parquet::schema::NodePtr> Apply(
      const ::parquet::schema::NodePtr& original_node, const NameMapping& name_mapping);

  // ParquetTypeVisitor interface implementation
  [[nodiscard]] Result<::parquet::schema::NodePtr> Message(
      const ::parquet::schema::GroupNode& message,
      std::vector<::parquet::schema::NodePtr>&& fields) override;

  [[nodiscard]] Result<::parquet::schema::NodePtr> Struct(
      const ::parquet::schema::GroupNode& struct_node,
      std::vector<::parquet::schema::NodePtr>&& fields) override;

  [[nodiscard]] Result<::parquet::schema::NodePtr> List(
      const ::parquet::schema::GroupNode& list_node,
      ::parquet::schema::NodePtr&& element) override;

  [[nodiscard]] Result<::parquet::schema::NodePtr> Map(
      const ::parquet::schema::GroupNode& map_node,
      std::optional<::parquet::schema::NodePtr>&& key,
      std::optional<::parquet::schema::NodePtr>&& value) override;

  [[nodiscard]] Result<::parquet::schema::NodePtr> Primitive(
      const ::parquet::schema::PrimitiveNode& primitive) override;

  // Override name normalization methods for List/Map handling
  void BeforeField(const ::parquet::schema::Node& field) override;
  void AfterField(const ::parquet::schema::Node& field) override;
  void BeforeElementField(const ::parquet::schema::Node& element) override;
  void AfterElementField(const ::parquet::schema::Node& element) override;
  void BeforeKeyField(const ::parquet::schema::Node& key) override;
  void AfterKeyField(const ::parquet::schema::Node& key) override;
  void BeforeValueField(const ::parquet::schema::Node& value) override;
  void AfterValueField(const ::parquet::schema::Node& value) override;
  void BeforeRepeatedElement(const ::parquet::schema::Node& element) override;
  void AfterRepeatedElement(const ::parquet::schema::Node& element) override;
  void BeforeRepeatedKeyValue(const ::parquet::schema::Node& key_value) override;
  void AfterRepeatedKeyValue(const ::parquet::schema::Node& key_value) override;

 private:
  /// \brief Find a mapped field for the current path.
  ///
  /// \param path The field path to look up.
  /// \return The mapped field if found, nullopt otherwise.
  [[nodiscard]] std::optional<MappedFieldConstRef> FindMappedField(
      const std::vector<std::string>& path) const;

  /// \brief Get the current field path.
  ///
  /// \return The current field path as a vector of strings.
  [[nodiscard]] std::vector<std::string> CurrentPath() const;

  /// \brief Create a new node with the specified field ID.
  ///
  /// \param original_node The original node.
  /// \param field_id The field ID to assign.
  /// \return A new node with the field ID set.
  [[nodiscard]] static ::parquet::schema::NodePtr CreateNodeWithFieldId(
      const ::parquet::schema::NodePtr& original_node, int32_t field_id);

  /// \brief Filter out null nodes from a vector.
  ///
  /// \param nodes The vector of nodes to filter.
  /// \return A new vector with null nodes removed.
  [[nodiscard]] static std::vector<::parquet::schema::NodePtr> FilterNullNodes(
      std::vector<::parquet::schema::NodePtr>&& nodes);

  /// \brief Determine the list element type from a list node.
  ///
  /// \param list_node The list node.
  /// \return The list element type.
  [[nodiscard]] static ::parquet::schema::NodePtr DetermineListElementType(
      const ::parquet::schema::GroupNode& list_node);

  const NameMapping& name_mapping_;
  std::deque<std::string> field_names_;
};

}  // namespace iceberg::parquet