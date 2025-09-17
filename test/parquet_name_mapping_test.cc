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

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/json/from_string.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>
#include <parquet/schema.h>

#include "iceberg/arrow/arrow_fs_file_io_internal.h"
#include "iceberg/constants.h"
#include "iceberg/file_format.h"
#include "iceberg/manifest_entry.h"
#include "iceberg/name_mapping.h"
#include "iceberg/parquet/apply_name_mapping_visitor_internal.h"
#include "iceberg/parquet/parquet_register.h"
#include "iceberg/parquet/parquet_schema_util_internal.h"
#include "iceberg/schema.h"
#include "iceberg/snapshot.h"
#include "iceberg/table_metadata.h"
#include "iceberg/table_scan.h"
#include "iceberg/type.h"
#include "iceberg/util/checked_cast.h"
#include "matchers.h"
#include "temp_file_test_base.h"

namespace iceberg::parquet {

class ParquetNameMappingTest : public TempFileTestBase {
 protected:
  static void SetUpTestSuite() { parquet::RegisterAll(); }

  void SetUp() override {
    TempFileTestBase::SetUp();
    file_io_ = arrow::ArrowFileSystemFileIO::MakeLocalFileIO();
    temp_parquet_file_ = CreateNewTempFilePathWithSuffix(".parquet");
  }

  // Helper to create a simple name mapping for testing
  std::shared_ptr<NameMapping> CreateSimpleNameMapping() {
    std::vector<MappedField> fields;
    fields.emplace_back(MappedField{.names = {"id"}, .field_id = 1});
    fields.emplace_back(MappedField{.names = {"name"}, .field_id = 2});
    fields.emplace_back(MappedField{.names = {"age"}, .field_id = 3});
    return NameMapping::Make(std::move(fields));
  }

  // Helper to create a nested name mapping for testing
  std::shared_ptr<NameMapping> CreateNestedNameMapping() {
    std::vector<MappedField> fields;
    fields.emplace_back(MappedField{.names = {"id"}, .field_id = 1});

    // Nested struct mapping
    std::vector<MappedField> address_fields;
    address_fields.emplace_back(MappedField{.names = {"street"}, .field_id = 101});
    address_fields.emplace_back(MappedField{.names = {"city"}, .field_id = 102});
    auto address_mapping = MappedFields::Make(std::move(address_fields));

    fields.emplace_back(MappedField{.names = {"address"},
                                    .field_id = 2,
                                    .nested_mapping = std::move(address_mapping)});

    return NameMapping::Make(std::move(fields));
  }

  // Helper to create a Parquet schema node without field IDs
  ::parquet::schema::NodePtr CreateSchemaWithoutFieldIds() {
    auto id_node = ::parquet::schema::PrimitiveNode::Make(
        "id", ::parquet::Repetition::REQUIRED, ::parquet::LogicalType::None(),
        ::parquet::Type::INT32, -1, -1);
    auto name_node = ::parquet::schema::PrimitiveNode::Make(
        "name", ::parquet::Repetition::OPTIONAL, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);
    auto age_node = ::parquet::schema::PrimitiveNode::Make(
        "age", ::parquet::Repetition::OPTIONAL, ::parquet::LogicalType::None(),
        ::parquet::Type::INT32, -1, -1);

    return ::parquet::schema::GroupNode::Make("schema", ::parquet::Repetition::REQUIRED,
                                              {id_node, name_node, age_node}, nullptr,
                                              -1);
  }

  // Helper to create a nested Parquet schema without field IDs
  ::parquet::schema::NodePtr CreateNestedSchemaWithoutFieldIds() {
    auto id_node = ::parquet::schema::PrimitiveNode::Make(
        "id", ::parquet::Repetition::REQUIRED, ::parquet::LogicalType::None(),
        ::parquet::Type::INT32, -1, -1);

    auto street_node = ::parquet::schema::PrimitiveNode::Make(
        "street", ::parquet::Repetition::OPTIONAL, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);
    auto city_node = ::parquet::schema::PrimitiveNode::Make(
        "city", ::parquet::Repetition::OPTIONAL, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);

    auto address_node =
        ::parquet::schema::GroupNode::Make("address", ::parquet::Repetition::OPTIONAL,
                                           {street_node, city_node}, nullptr, -1);

    return ::parquet::schema::GroupNode::Make("schema", ::parquet::Repetition::REQUIRED,
                                              {id_node, address_node}, nullptr, -1);
  }

  // Helper to create a List schema without field IDs for testing
  ::parquet::schema::NodePtr CreateListSchemaWithoutFieldIds() {
    auto element_node = ::parquet::schema::PrimitiveNode::Make(
        "element", ::parquet::Repetition::REPEATED, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);

    auto list_node = ::parquet::schema::GroupNode::Make(
        "items", ::parquet::Repetition::OPTIONAL, {element_node},
        ::parquet::LogicalType::List(), -1);

    auto id_node = ::parquet::schema::PrimitiveNode::Make(
        "id", ::parquet::Repetition::REQUIRED, ::parquet::LogicalType::None(),
        ::parquet::Type::INT32, -1, -1);

    return ::parquet::schema::GroupNode::Make("schema", ::parquet::Repetition::REQUIRED,
                                              {id_node, list_node}, nullptr, -1);
  }

  // Helper to create a Map schema without field IDs for testing
  ::parquet::schema::NodePtr CreateMapSchemaWithoutFieldIds() {
    auto key_node = ::parquet::schema::PrimitiveNode::Make(
        "key", ::parquet::Repetition::REQUIRED, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);
    auto value_node = ::parquet::schema::PrimitiveNode::Make(
        "value", ::parquet::Repetition::OPTIONAL, ::parquet::LogicalType::String(),
        ::parquet::Type::BYTE_ARRAY, -1, -1);

    auto key_value_node =
        ::parquet::schema::GroupNode::Make("key_value", ::parquet::Repetition::REPEATED,
                                           {key_node, value_node}, nullptr, -1);

    auto map_node = ::parquet::schema::GroupNode::Make(
        "data", ::parquet::Repetition::OPTIONAL, {key_value_node},
        ::parquet::LogicalType::Map(), -1);

    auto id_node = ::parquet::schema::PrimitiveNode::Make(
        "id", ::parquet::Repetition::REQUIRED, ::parquet::LogicalType::None(),
        ::parquet::Type::INT32, -1, -1);

    return ::parquet::schema::GroupNode::Make("schema", ::parquet::Repetition::REQUIRED,
                                              {id_node, map_node}, nullptr, -1);
  }

  // Helper to create a List name mapping for testing
  std::shared_ptr<NameMapping> CreateListNameMapping() {
    std::vector<MappedField> fields;
    fields.emplace_back(MappedField{.names = {"id"}, .field_id = 1});

    // List mapping with element mapping
    std::vector<MappedField> element_fields;
    element_fields.emplace_back(MappedField{.names = {"element"}, .field_id = 101});
    auto element_mapping = MappedFields::Make(std::move(element_fields));

    fields.emplace_back(MappedField{
        .names = {"items"}, .field_id = 2, .nested_mapping = std::move(element_mapping)});

    return NameMapping::Make(std::move(fields));
  }

  // Helper to create a Map name mapping for testing
  std::shared_ptr<NameMapping> CreateMapNameMapping() {
    std::vector<MappedField> fields;
    fields.emplace_back(MappedField{.names = {"id"}, .field_id = 1});

    // Map mapping with key and value mappings
    std::vector<MappedField> map_fields;
    map_fields.emplace_back(MappedField{.names = {"key"}, .field_id = 101});
    map_fields.emplace_back(MappedField{.names = {"value"}, .field_id = 102});
    auto map_mapping = MappedFields::Make(std::move(map_fields));

    fields.emplace_back(MappedField{
        .names = {"data"}, .field_id = 2, .nested_mapping = std::move(map_mapping)});

    return NameMapping::Make(std::move(fields));
  }
  void CreateParquetFileWithoutFieldIds() {
    auto arrow_schema = ::arrow::schema({::arrow::field("id", ::arrow::int32(), false),
                                         ::arrow::field("name", ::arrow::utf8(), true),
                                         ::arrow::field("age", ::arrow::int32(), true)});

    auto table = ::arrow::Table::FromRecordBatches(
                     arrow_schema,
                     {::arrow::RecordBatch::FromStructArray(
                          ::arrow::json::ArrayFromJSONString(
                              ::arrow::struct_(arrow_schema->fields()),
                              R"([[1, "Alice", 25], [2, "Bob", 30], [3, "Charlie", 35]])")
                              .ValueOrDie())
                          .ValueOrDie()})
                     .ValueOrDie();

    auto io = internal::checked_cast<arrow::ArrowFileSystemFileIO&>(*file_io_);
    auto outfile = io.fs()->OpenOutputStream(temp_parquet_file_).ValueOrDie();

    ASSERT_TRUE(::parquet::arrow::WriteTable(*table, ::arrow::default_memory_pool(),
                                             outfile, 1024)
                    .ok());
  }

  std::shared_ptr<FileIO> file_io_;
  std::string temp_parquet_file_;
};

TEST_F(ParquetNameMappingTest, MakeParquetNodeWithFieldIds_SimpleSchema) {
  auto schema = CreateSchemaWithoutFieldIds();
  auto name_mapping = CreateSimpleNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = MakeParquetNodeWithFieldIds(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  // Verify the root node doesn't have a field ID (as expected for root)
  EXPECT_EQ(enhanced_schema->field_id(), -1);

  // Verify child nodes have the correct field IDs
  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 3);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // name
  EXPECT_EQ(group_node->field(2)->field_id(), 3);  // age
}

TEST_F(ParquetNameMappingTest, MakeParquetNodeWithFieldIds_NestedSchema) {
  auto schema = CreateNestedSchemaWithoutFieldIds();
  auto name_mapping = CreateNestedNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = MakeParquetNodeWithFieldIds(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 2);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // address

  // Check nested fields
  auto address_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(group_node->field(1));
  EXPECT_EQ(address_node->field_count(), 2);
  EXPECT_EQ(address_node->field(0)->field_id(), 101);  // street
  EXPECT_EQ(address_node->field(1)->field_id(), 102);  // city
}

TEST_F(ParquetNameMappingTest, ApplyNameMappingVisitor_SimpleSchema) {
  auto schema = CreateSchemaWithoutFieldIds();
  auto name_mapping = CreateSimpleNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = ApplyNameMappingVisitor::Apply(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  // Verify the root node doesn't have a field ID (as expected for root)
  EXPECT_EQ(enhanced_schema->field_id(), -1);

  // Verify child nodes have the correct field IDs
  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 3);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // name
  EXPECT_EQ(group_node->field(2)->field_id(), 3);  // age
}

TEST_F(ParquetNameMappingTest, ApplyNameMappingVisitor_NestedSchema) {
  auto schema = CreateNestedSchemaWithoutFieldIds();
  auto name_mapping = CreateNestedNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = ApplyNameMappingVisitor::Apply(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 2);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // address

  // Check nested fields
  auto address_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(group_node->field(1));
  EXPECT_EQ(address_node->field_count(), 2);
  EXPECT_EQ(address_node->field(0)->field_id(), 101);  // street
  EXPECT_EQ(address_node->field(1)->field_id(), 102);  // city
}

TEST_F(ParquetNameMappingTest, ApplyNameMappingVisitor_PartialMapping) {
  auto schema = CreateSchemaWithoutFieldIds();

  // Create name mapping that's missing the "age" field
  std::vector<MappedField> fields;
  fields.emplace_back(MappedField{.names = {"id"}, .field_id = 1});
  fields.emplace_back(MappedField{.names = {"name"}, .field_id = 2});
  auto incomplete_mapping = NameMapping::Make(std::move(fields));

  auto result = ApplyNameMappingVisitor::Apply(schema, *incomplete_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;

  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 3);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);   // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);   // name
  EXPECT_EQ(group_node->field(2)->field_id(), -1);  // age (no mapping)
}

TEST_F(ParquetNameMappingTest, ApplyNameMappingVisitor_ListSchema) {
  auto schema = CreateListSchemaWithoutFieldIds();
  auto name_mapping = CreateListNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = ApplyNameMappingVisitor::Apply(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 2);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // items (list)

  // Check list element field
  auto list_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(group_node->field(1));
  EXPECT_EQ(list_node->field_count(), 1);
  EXPECT_EQ(list_node->field(0)->field_id(), 101);  // element
}

TEST_F(ParquetNameMappingTest, ApplyNameMappingVisitor_MapSchema) {
  auto schema = CreateMapSchemaWithoutFieldIds();
  auto name_mapping = CreateMapNameMapping();

  ASSERT_FALSE(HasFieldIds(schema));

  auto result = ApplyNameMappingVisitor::Apply(schema, *name_mapping);
  ASSERT_THAT(result, IsOk());

  const auto& enhanced_schema = *result;
  EXPECT_TRUE(HasFieldIds(enhanced_schema));

  auto group_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(enhanced_schema);
  EXPECT_EQ(group_node->field_count(), 2);
  EXPECT_EQ(group_node->field(0)->field_id(), 1);  // id
  EXPECT_EQ(group_node->field(1)->field_id(), 2);  // data (map)

  // Check map key-value fields
  auto map_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(group_node->field(1));
  EXPECT_EQ(map_node->field_count(), 1);

  auto key_value_node =
      std::static_pointer_cast<::parquet::schema::GroupNode>(map_node->field(0));
  EXPECT_EQ(key_value_node->field_count(), 2);
  EXPECT_EQ(key_value_node->field(0)->field_id(), 101);  // key
  EXPECT_EQ(key_value_node->field(1)->field_id(), 102);  // value
}

TEST_F(ParquetNameMappingTest, ParquetReader_WithNameMapping_Integration) {
  CreateParquetFileWithoutFieldIds();

  auto data_file = std::make_shared<DataFile>();
  data_file->file_path = temp_parquet_file_;
  data_file->file_format = FileFormatType::kParquet;

  auto projected_schema = std::make_shared<Schema>(
      std::vector<SchemaField>{SchemaField::MakeRequired(1, "id", int32()),
                               SchemaField::MakeOptional(2, "name", string()),
                               SchemaField::MakeOptional(3, "age", int32())});

  auto name_mapping = CreateSimpleNameMapping();
  FileScanTask task(data_file);

  // Test the new ToArrow method with name mapping
  auto stream_result = task.ToArrow(file_io_, projected_schema, nullptr, name_mapping);
  ASSERT_THAT(stream_result, IsOk());

  auto stream = std::move(stream_result.value());
  auto record_batch_reader = ::arrow::ImportRecordBatchReader(&stream).ValueOrDie();

  auto result = record_batch_reader->Next();
  ASSERT_TRUE(result.ok()) << result.status().message();
  auto actual_batch = result.ValueOrDie();
  ASSERT_NE(actual_batch, nullptr);

  // Verify the data can be read correctly
  EXPECT_EQ(actual_batch->num_rows(), 3);
  EXPECT_EQ(actual_batch->num_columns(), 3);
}

TEST_F(ParquetNameMappingTest, ParquetReader_WithoutNameMapping_Fails) {
  CreateParquetFileWithoutFieldIds();

  auto data_file = std::make_shared<DataFile>();
  data_file->file_path = temp_parquet_file_;
  data_file->file_format = FileFormatType::kParquet;

  auto projected_schema = std::make_shared<Schema>(
      std::vector<SchemaField>{SchemaField::MakeRequired(1, "id", int32()),
                               SchemaField::MakeOptional(2, "name", string())});

  FileScanTask task(data_file);

  // This should fail because the Parquet file has no field IDs and no name mapping
  auto stream_result = task.ToArrow(file_io_, projected_schema, nullptr, nullptr);
  ASSERT_THAT(stream_result, IsError(ErrorKind::kInvalidArgument));
  ASSERT_THAT(
      stream_result,
      HasErrorMessage("Parquet file has no field IDs and no name mapping provided"));
}

// Test name mapping integration with TableScan
class TableScanNameMappingTest : public ::testing::Test {
 protected:
  void SetUp() override { file_io_ = arrow::ArrowFileSystemFileIO::MakeLocalFileIO(); }

  std::shared_ptr<FileIO> file_io_;
};

TEST_F(TableScanNameMappingTest, NameMappingConstantDefined) {
  // Simple test to verify the constant is defined correctly
  std::string name_mapping_key(kNameMappingProperty);
  EXPECT_EQ(name_mapping_key, "schema.name-mapping.default");
}

}  // namespace iceberg::parquet
