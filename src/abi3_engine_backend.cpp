// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#define Py_LIMITED_API 0x030A0000
#include <Python.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log_utils.h"
#include "index/common_structs.h"
#include "index/index_engine.h"
#include "store/kv_store.h"
#include "store/persist_store.h"
#include "store/volatile_store.h"

namespace vdb = vectordb;

namespace {

constexpr const char* kIndexCapsuleName = "openviking.vectordb.IndexEngine";
constexpr const char* kStoreCapsuleName = "openviking.vectordb.KVStore";

void raise_type_error(const std::string& message) {
  PyErr_SetString(PyExc_TypeError, message.c_str());
}

void raise_value_error(const std::string& message) {
  PyErr_SetString(PyExc_ValueError, message.c_str());
}

void raise_runtime_error(const std::string& message) {
  PyErr_SetString(PyExc_RuntimeError, message.c_str());
}

bool py_to_string(PyObject* obj, std::string* out, bool allow_bytes) {
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (data == nullptr) {
      return false;
    }
    out->assign(data, static_cast<size_t>(size));
    return true;
  }

  if (allow_bytes && PyBytes_Check(obj)) {
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(obj, &data, &size) < 0) {
      return false;
    }
    out->assign(data, static_cast<size_t>(size));
    return true;
  }

  raise_type_error("Expected str" + std::string(allow_bytes ? " or bytes" : ""));
  return false;
}

bool py_to_uint64(PyObject* obj, uint64_t* out) {
  const unsigned long long value = PyLong_AsUnsignedLongLong(obj);
  if (PyErr_Occurred() != nullptr) {
    return false;
  }
  *out = static_cast<uint64_t>(value);
  return true;
}

bool py_to_uint32(PyObject* obj, uint32_t* out) {
  const unsigned long value = PyLong_AsUnsignedLong(obj);
  if (PyErr_Occurred() != nullptr) {
    return false;
  }
  *out = static_cast<uint32_t>(value);
  return true;
}

bool py_to_float(PyObject* obj, float* out) {
  const double value = PyFloat_AsDouble(obj);
  if (PyErr_Occurred() != nullptr) {
    return false;
  }
  *out = static_cast<float>(value);
  return true;
}

bool py_to_float_vector(PyObject* obj, std::vector<float>* out) {
  const Py_ssize_t size = PySequence_Size(obj);
  if (size < 0) {
    raise_type_error("Expected a sequence of floats");
    return false;
  }

  out->clear();
  out->reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_GetItem(obj, i);
    if (item == nullptr) {
      return false;
    }
    float value = 0.0f;
    const bool ok = py_to_float(item, &value);
    Py_DECREF(item);
    if (!ok) {
      return false;
    }
    out->push_back(value);
  }
  return true;
}

bool py_to_string_vector(PyObject* obj, std::vector<std::string>* out) {
  const Py_ssize_t size = PySequence_Size(obj);
  if (size < 0) {
    raise_type_error("Expected a sequence of strings");
    return false;
  }

  out->clear();
  out->reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_GetItem(obj, i);
    if (item == nullptr) {
      return false;
    }
    std::string value;
    const bool ok = py_to_string(item, &value, false);
    Py_DECREF(item);
    if (!ok) {
      return false;
    }
    out->push_back(std::move(value));
  }
  return true;
}

PyObject* float_vector_to_py(const std::vector<float>& values) {
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (list == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); ++i) {
    PyObject* item = PyFloat_FromDouble(values[static_cast<size_t>(i)]);
    if (item == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SetItem(list, i, item);
  }
  return list;
}

PyObject* string_vector_to_py(const std::vector<std::string>& values) {
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (list == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); ++i) {
    const auto& value = values[static_cast<size_t>(i)];
    PyObject* item = PyUnicode_FromStringAndSize(value.data(), static_cast<Py_ssize_t>(value.size()));
    if (item == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SetItem(list, i, item);
  }
  return list;
}

PyObject* uint64_vector_to_py(const std::vector<uint64_t>& values) {
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (list == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); ++i) {
    PyObject* item = PyLong_FromUnsignedLongLong(values[static_cast<size_t>(i)]);
    if (item == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SetItem(list, i, item);
  }
  return list;
}

template <typename T>
T* capsule_to_ptr(PyObject* capsule, const char* capsule_name) {
  if (!PyCapsule_CheckExact(capsule)) {
    raise_type_error("Expected capsule handle");
    return nullptr;
  }
  return static_cast<T*>(PyCapsule_GetPointer(capsule, capsule_name));
}

void index_capsule_destructor(PyObject* capsule) {
  auto* ptr = static_cast<vdb::IndexEngine*>(PyCapsule_GetPointer(capsule, kIndexCapsuleName));
  delete ptr;
}

void store_capsule_destructor(PyObject* capsule) {
  auto* ptr = static_cast<vdb::KVStore*>(PyCapsule_GetPointer(capsule, kStoreCapsuleName));
  delete ptr;
}

bool parse_add_request(PyObject* obj, vdb::AddDataRequest* request) {
  if (!PyDict_Check(obj)) {
    raise_type_error("Each add-data request must be a dict");
    return false;
  }

  PyObject* label = PyDict_GetItemString(obj, "label");
  if (label != nullptr && !py_to_uint64(label, &request->label)) {
    return false;
  }

  PyObject* vector = PyDict_GetItemString(obj, "vector");
  if (vector != nullptr && !py_to_float_vector(vector, &request->vector)) {
    return false;
  }

  PyObject* sparse_terms = PyDict_GetItemString(obj, "sparse_raw_terms");
  if (sparse_terms != nullptr &&
      !py_to_string_vector(sparse_terms, &request->sparse_raw_terms)) {
    return false;
  }

  PyObject* sparse_values = PyDict_GetItemString(obj, "sparse_values");
  if (sparse_values != nullptr &&
      !py_to_float_vector(sparse_values, &request->sparse_values)) {
    return false;
  }

  PyObject* fields_str = PyDict_GetItemString(obj, "fields_str");
  if (fields_str != nullptr &&
      !py_to_string(fields_str, &request->fields_str, true)) {
    return false;
  }

  PyObject* old_fields_str = PyDict_GetItemString(obj, "old_fields_str");
  if (old_fields_str != nullptr &&
      !py_to_string(old_fields_str, &request->old_fields_str, true)) {
    return false;
  }

  return true;
}

bool parse_delete_request(PyObject* obj, vdb::DeleteDataRequest* request) {
  if (!PyDict_Check(obj)) {
    raise_type_error("Each delete-data request must be a dict");
    return false;
  }

  PyObject* label = PyDict_GetItemString(obj, "label");
  if (label != nullptr && !py_to_uint64(label, &request->label)) {
    return false;
  }

  PyObject* old_fields_str = PyDict_GetItemString(obj, "old_fields_str");
  if (old_fields_str != nullptr &&
      !py_to_string(old_fields_str, &request->old_fields_str, true)) {
    return false;
  }

  return true;
}

bool parse_search_request(PyObject* obj, vdb::SearchRequest* request) {
  if (!PyDict_Check(obj)) {
    raise_type_error("Search request must be a dict");
    return false;
  }

  PyObject* query = PyDict_GetItemString(obj, "query");
  if (query != nullptr && !py_to_float_vector(query, &request->query)) {
    return false;
  }

  PyObject* sparse_terms = PyDict_GetItemString(obj, "sparse_raw_terms");
  if (sparse_terms != nullptr &&
      !py_to_string_vector(sparse_terms, &request->sparse_raw_terms)) {
    return false;
  }

  PyObject* sparse_values = PyDict_GetItemString(obj, "sparse_values");
  if (sparse_values != nullptr &&
      !py_to_float_vector(sparse_values, &request->sparse_values)) {
    return false;
  }

  PyObject* topk = PyDict_GetItemString(obj, "topk");
  if (topk != nullptr && !py_to_uint32(topk, &request->topk)) {
    return false;
  }

  PyObject* dsl = PyDict_GetItemString(obj, "dsl");
  if (dsl != nullptr && !py_to_string(dsl, &request->dsl, false)) {
    return false;
  }

  return true;
}

bool parse_storage_op(PyObject* obj, vdb::StorageOp* op) {
  if (!PyDict_Check(obj)) {
    raise_type_error("Each storage op must be a dict");
    return false;
  }

  PyObject* type = PyDict_GetItemString(obj, "type");
  const long type_value = type == nullptr ? 0 : PyLong_AsLong(type);
  if (type != nullptr && PyErr_Occurred() != nullptr) {
    return false;
  }
  op->type = type_value == static_cast<long>(vdb::StorageOp::DELETE_OP)
      ? vdb::StorageOp::DELETE_OP
      : vdb::StorageOp::PUT_OP;

  PyObject* key = PyDict_GetItemString(obj, "key");
  if (key == nullptr || !py_to_string(key, &op->key, false)) {
    return false;
  }

  PyObject* value = PyDict_GetItemString(obj, "value");
  if (value != nullptr && !py_to_string(value, &op->value, true)) {
    return false;
  }

  return true;
}

template <typename RequestT>
bool parse_request_list(PyObject* obj,
                        bool (*parse_item)(PyObject*, RequestT*),
                        std::vector<RequestT>* out) {
  const Py_ssize_t size = PySequence_Size(obj);
  if (size < 0) {
    raise_type_error("Expected a sequence");
    return false;
  }

  out->clear();
  out->reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_GetItem(obj, i);
    if (item == nullptr) {
      return false;
    }
    RequestT request;
    const bool ok = parse_item(item, &request);
    Py_DECREF(item);
    if (!ok) {
      return false;
    }
    out->push_back(std::move(request));
  }

  return true;
}

bool parse_string_list(PyObject* obj, std::vector<std::string>* out) {
  return py_to_string_vector(obj, out);
}

bool parse_binary_list(PyObject* obj, std::vector<std::string>* out) {
  const Py_ssize_t size = PySequence_Size(obj);
  if (size < 0) {
    raise_type_error("Expected a sequence of bytes");
    return false;
  }
  out->clear();
  out->reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_GetItem(obj, i);
    if (item == nullptr) {
      return false;
    }
    std::string value;
    const bool ok = py_to_string(item, &value, true);
    Py_DECREF(item);
    if (!ok) {
      return false;
    }
    out->push_back(std::move(value));
  }
  return true;
}

PyObject* build_search_result(const vdb::SearchResult& result) {
  PyObject* payload = PyDict_New();
  if (payload == nullptr) {
    return nullptr;
  }

  PyObject* result_num = PyLong_FromUnsignedLong(result.result_num);
  PyObject* labels = uint64_vector_to_py(result.labels);
  PyObject* scores = float_vector_to_py(result.scores);
  PyObject* extra_json = PyUnicode_FromStringAndSize(
      result.extra_json.data(), static_cast<Py_ssize_t>(result.extra_json.size()));
  if (result_num == nullptr || labels == nullptr || scores == nullptr ||
      extra_json == nullptr) {
    Py_XDECREF(result_num);
    Py_XDECREF(labels);
    Py_XDECREF(scores);
    Py_XDECREF(extra_json);
    Py_DECREF(payload);
    return nullptr;
  }

  PyDict_SetItemString(payload, "result_num", result_num);
  PyDict_SetItemString(payload, "labels", labels);
  PyDict_SetItemString(payload, "scores", scores);
  PyDict_SetItemString(payload, "extra_json", extra_json);
  Py_DECREF(result_num);
  Py_DECREF(labels);
  Py_DECREF(scores);
  Py_DECREF(extra_json);
  return payload;
}

PyObject* build_state_result(const vdb::StateResult& result) {
  PyObject* payload = PyDict_New();
  if (payload == nullptr) {
    return nullptr;
  }

  PyObject* update_ts = PyLong_FromUnsignedLongLong(result.update_timestamp);
  PyObject* element_count = PyLong_FromUnsignedLongLong(result.element_count);
  if (update_ts == nullptr || element_count == nullptr) {
    Py_XDECREF(update_ts);
    Py_XDECREF(element_count);
    Py_DECREF(payload);
    return nullptr;
  }

  PyDict_SetItemString(payload, "update_timestamp", update_ts);
  PyDict_SetItemString(payload, "element_count", element_count);
  Py_DECREF(update_ts);
  Py_DECREF(element_count);
  return payload;
}

PyObject* py_init_logging(PyObject*, PyObject* args, PyObject* kwargs) {
  const char* log_level = nullptr;
  const char* log_output = nullptr;
  const char* log_format = "[%Y-%m-%d %H:%M:%S.%e] [%l] %v";
  static const char* keywords[] = {"log_level", "log_output", "log_format", nullptr};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|s", const_cast<char**>(keywords),
                                   &log_level, &log_output, &log_format)) {
    return nullptr;
  }

  try {
    vdb::init_logging(log_level, log_output, log_format);
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }

  Py_RETURN_NONE;
}

PyObject* py_new_index_engine(PyObject*, PyObject* args) {
  const char* path_or_json = nullptr;
  if (!PyArg_ParseTuple(args, "s", &path_or_json)) {
    return nullptr;
  }

  try {
    return PyCapsule_New(new vdb::IndexEngine(path_or_json), kIndexCapsuleName,
                         index_capsule_destructor);
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_index_engine_add_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* items = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &items)) {
    return nullptr;
  }

  auto* engine = capsule_to_ptr<vdb::IndexEngine>(capsule, kIndexCapsuleName);
  if (engine == nullptr) {
    return nullptr;
  }

  std::vector<vdb::AddDataRequest> requests;
  if (!parse_request_list(items, parse_add_request, &requests)) {
    return nullptr;
  }

  try {
    return PyLong_FromLong(engine->add_data(requests));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_index_engine_delete_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* items = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &items)) {
    return nullptr;
  }

  auto* engine = capsule_to_ptr<vdb::IndexEngine>(capsule, kIndexCapsuleName);
  if (engine == nullptr) {
    return nullptr;
  }

  std::vector<vdb::DeleteDataRequest> requests;
  if (!parse_request_list(items, parse_delete_request, &requests)) {
    return nullptr;
  }

  try {
    return PyLong_FromLong(engine->delete_data(requests));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_index_engine_search(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* request_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &request_obj)) {
    return nullptr;
  }

  auto* engine = capsule_to_ptr<vdb::IndexEngine>(capsule, kIndexCapsuleName);
  if (engine == nullptr) {
    return nullptr;
  }

  vdb::SearchRequest request;
  if (!parse_search_request(request_obj, &request)) {
    return nullptr;
  }

  try {
    return build_search_result(engine->search(request));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_index_engine_dump(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  const char* path = nullptr;
  if (!PyArg_ParseTuple(args, "Os", &capsule, &path)) {
    return nullptr;
  }

  auto* engine = capsule_to_ptr<vdb::IndexEngine>(capsule, kIndexCapsuleName);
  if (engine == nullptr) {
    return nullptr;
  }

  try {
    return PyLong_FromLongLong(engine->dump(path));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_index_engine_get_state(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }

  auto* engine = capsule_to_ptr<vdb::IndexEngine>(capsule, kIndexCapsuleName);
  if (engine == nullptr) {
    return nullptr;
  }

  try {
    return build_state_result(engine->get_state());
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_new_persist_store(PyObject*, PyObject* args) {
  const char* path = nullptr;
  if (!PyArg_ParseTuple(args, "s", &path)) {
    return nullptr;
  }

  try {
    return PyCapsule_New(new vdb::PersistStore(path), kStoreCapsuleName,
                         store_capsule_destructor);
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_new_volatile_store(PyObject*, PyObject*) {
  try {
    return PyCapsule_New(new vdb::VolatileStore(), kStoreCapsuleName,
                         store_capsule_destructor);
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_exec_op(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* ops_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &ops_obj)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  std::vector<vdb::StorageOp> ops;
  if (!parse_request_list(ops_obj, parse_storage_op, &ops)) {
    return nullptr;
  }

  try {
    return PyLong_FromLong(store->exec_op(ops));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_get_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* keys_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &keys_obj)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  std::vector<std::string> keys;
  if (!parse_string_list(keys_obj, &keys)) {
    return nullptr;
  }

  try {
    const auto values = store->get_data(keys);
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (list == nullptr) {
      return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(values.size()); ++i) {
      const auto& value = values[static_cast<size_t>(i)];
      PyObject* item = PyBytes_FromStringAndSize(value.data(),
                                                 static_cast<Py_ssize_t>(value.size()));
      if (item == nullptr) {
        Py_DECREF(list);
        return nullptr;
      }
      PyList_SetItem(list, i, item);
    }
    return list;
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_put_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* keys_obj = nullptr;
  PyObject* values_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OOO", &capsule, &keys_obj, &values_obj)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  std::vector<std::string> keys;
  std::vector<std::string> values;
  if (!parse_string_list(keys_obj, &keys) || !parse_binary_list(values_obj, &values)) {
    return nullptr;
  }
  if (keys.size() != values.size()) {
    raise_value_error("keys and values must have the same length");
    return nullptr;
  }

  try {
    return PyLong_FromLong(store->put_data(keys, values));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_delete_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* keys_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &keys_obj)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  std::vector<std::string> keys;
  if (!parse_string_list(keys_obj, &keys)) {
    return nullptr;
  }

  try {
    return PyLong_FromLong(store->delete_data(keys));
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_clear_data(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  try {
    return PyLong_FromLong(store->clear_data());
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyObject* py_store_seek_range(PyObject*, PyObject* args) {
  PyObject* capsule = nullptr;
  const char* start_key = nullptr;
  const char* end_key = nullptr;
  if (!PyArg_ParseTuple(args, "Oss", &capsule, &start_key, &end_key)) {
    return nullptr;
  }

  auto* store = capsule_to_ptr<vdb::KVStore>(capsule, kStoreCapsuleName);
  if (store == nullptr) {
    return nullptr;
  }

  try {
    const auto items = store->seek_range(start_key, end_key);
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(items.size()));
    if (list == nullptr) {
      return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(items.size()); ++i) {
      const auto& item = items[static_cast<size_t>(i)];
      PyObject* tuple = PyTuple_New(2);
      PyObject* key = PyUnicode_FromStringAndSize(item.first.data(),
                                                  static_cast<Py_ssize_t>(item.first.size()));
      PyObject* value = PyBytes_FromStringAndSize(item.second.data(),
                                                  static_cast<Py_ssize_t>(item.second.size()));
      if (tuple == nullptr || key == nullptr || value == nullptr) {
        Py_XDECREF(tuple);
        Py_XDECREF(key);
        Py_XDECREF(value);
        Py_DECREF(list);
        return nullptr;
      }
      PyTuple_SetItem(tuple, 0, key);
      PyTuple_SetItem(tuple, 1, value);
      PyList_SetItem(list, i, tuple);
    }
    return list;
  } catch (const std::exception& exc) {
    raise_runtime_error(exc.what());
    return nullptr;
  }
}

PyMethodDef kModuleMethods[] = {
    {"_init_logging", reinterpret_cast<PyCFunction>(py_init_logging),
     METH_VARARGS | METH_KEYWORDS, "Initialize vectordb logging."},
    {"_new_index_engine", py_new_index_engine, METH_VARARGS, "Create an index engine handle."},
    {"_index_engine_add_data", py_index_engine_add_data, METH_VARARGS,
     "Add data to the index engine."},
    {"_index_engine_delete_data", py_index_engine_delete_data, METH_VARARGS,
     "Delete data from the index engine."},
    {"_index_engine_search", py_index_engine_search, METH_VARARGS,
     "Search the index engine."},
    {"_index_engine_dump", py_index_engine_dump, METH_VARARGS, "Dump index state to disk."},
    {"_index_engine_get_state", py_index_engine_get_state, METH_VARARGS,
     "Read index engine state."},
    {"_new_persist_store", py_new_persist_store, METH_VARARGS, "Create a persistent store."},
    {"_new_volatile_store", py_new_volatile_store, METH_NOARGS, "Create a volatile store."},
    {"_store_exec_op", py_store_exec_op, METH_VARARGS, "Execute store operations."},
    {"_store_get_data", py_store_get_data, METH_VARARGS, "Read values from the store."},
    {"_store_put_data", py_store_put_data, METH_VARARGS, "Write values to the store."},
    {"_store_delete_data", py_store_delete_data, METH_VARARGS, "Delete keys from the store."},
    {"_store_clear_data", py_store_clear_data, METH_VARARGS, "Clear the store."},
    {"_store_seek_range", py_store_seek_range, METH_VARARGS,
     "Read a key range from the store."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_ov_engine_backend",
    "OpenViking abi3 vectordb backend.",
    -1,
    kModuleMethods,
};

}  // namespace

#ifndef OV_PY_MODULE_NAME
#define OV_PY_MODULE_NAME _native
#endif

#define OV_CONCAT_IMPL(a, b) a##b
#define OV_CONCAT(a, b) OV_CONCAT_IMPL(a, b)

PyMODINIT_FUNC OV_CONCAT(PyInit_, OV_PY_MODULE_NAME)(void) {
  PyObject* module = PyModule_Create(&kModuleDef);
  if (module == nullptr) {
    return nullptr;
  }

  if (PyModule_AddStringConstant(module, "_ENGINE_BACKEND_API", "abi3-v1") < 0) {
    Py_DECREF(module);
    return nullptr;
  }

  return module;
}
