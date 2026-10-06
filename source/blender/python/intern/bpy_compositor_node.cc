/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup pythonintern
 *
 * This file implements the Python side of the evaluation of Python registered compositor nodes.
 * The compositor calls into this file through the callbacks declared in NOD_composite_python.hh.
 *
 * Images on the CPU are exposed as `bpy_compositor.Buffer` objects, which only implement the
 * buffer protocol (to be used with `numpy.asarray` or `memoryview`) and a `shape` attribute.
 * Images on the GPU are exposed as `gpu.types.GPUTexture`.
 */

#include <Python.h>

#include <string>

#include "BLI_implicit_sharing.hh"
#include "BLI_span.hh"

#include "BKE_node.hh"
#include "BKE_node_runtime.hh"

#include "DNA_ID.h"
#include "DNA_node_types.h"

#include "RNA_access.hh"

#include "COM_result.hh"

#include "NOD_composite_python.hh"

#include "bpy_compositor_node.hh"
#include "bpy_rna.hh"

#include "../gpu/gpu_py_texture.hh"

namespace blender {

namespace compositor_python = nodes::compositor_python;
using compositor_python::EvalMode;
using compositor_python::SocketValue;

/* -------------------------------------------------------------------- */
/** \name Buffer Type
 * \{ */

struct BPyCompositorBuffer {
  PyObject_HEAD
  void *data;
  Py_ssize_t shape[3];
  Py_ssize_t strides[3];
  int ndim;
  char format[2];
  Py_ssize_t itemsize;
  bool readonly;
  /* A user is added on creation and removed on deallocation. Can be null. */
  const ImplicitSharingInfo *sharing;
};

/* The type is created at startup using #PyType_FromSpec and freed at exit. */
static PyObject *g_buffer_type = nullptr;

static void compositor_buffer_dealloc(PyObject *self_py)
{
  BPyCompositorBuffer *self = reinterpret_cast<BPyCompositorBuffer *>(self_py);
  if (self->sharing) {
    self->sharing->remove_user_and_delete_if_last();
  }
  /* Instances of heap types own a reference to their type. */
  PyTypeObject *type = Py_TYPE(self_py);
  type->tp_free(self_py);
  Py_DECREF(type);
}

static int compositor_buffer_getbuffer(PyObject *self_py, Py_buffer *view, int flags)
{
  BPyCompositorBuffer *self = reinterpret_cast<BPyCompositorBuffer *>(self_py);

  if (view == nullptr) {
    PyErr_SetString(PyExc_BufferError, "compositor buffer: view is NULL");
    return -1;
  }
  if ((flags & PyBUF_WRITABLE) && self->readonly) {
    PyErr_SetString(PyExc_BufferError, "compositor buffer: the buffer is read-only");
    return -1;
  }

  Py_ssize_t items_num = 1;
  for (int i = 0; i < self->ndim; i++) {
    items_num *= self->shape[i];
  }

  view->obj = Py_NewRef(self_py);
  view->buf = self->data;
  view->len = items_num * self->itemsize;
  view->readonly = self->readonly;
  view->itemsize = self->itemsize;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  if (flags & PyBUF_ND) {
    view->ndim = self->ndim;
    view->shape = self->shape;
    view->strides = (flags & PyBUF_STRIDES) ? self->strides : nullptr;
    view->format = (flags & PyBUF_FORMAT) ? self->format : nullptr;
  }
  else {
    /* Consumers that do not request a shape expect a flat array of bytes. */
    view->ndim = 1;
    view->shape = nullptr;
    view->strides = nullptr;
    view->format = nullptr;
    view->itemsize = 1;
  }
  return 0;
}

static PyObject *compositor_buffer_shape_get(PyObject *self_py, void * /*closure*/)
{
  BPyCompositorBuffer *self = reinterpret_cast<BPyCompositorBuffer *>(self_py);
  PyObject *shape = PyTuple_New(self->ndim);
  for (int i = 0; i < self->ndim; i++) {
    PyTuple_SET_ITEM(shape, i, PyLong_FromSsize_t(self->shape[i]));
  }
  return shape;
}

static PyGetSetDef compositor_buffer_getseters[] = {
    {"shape",
     compositor_buffer_shape_get,
     nullptr,
     "The shape of the buffer, (height, width, channels) or (height, width) if it has a single "
     "channel. Row 0 is at the bottom of the image.",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot compositor_buffer_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(compositor_buffer_dealloc)},
    {Py_bf_getbuffer, reinterpret_cast<void *>(compositor_buffer_getbuffer)},
    {Py_tp_getset, compositor_buffer_getseters},
    {Py_tp_doc,
     const_cast<char *>("Pixels of a compositor image, supports the buffer protocol "
                        "(use `numpy.asarray` to access them).")},
    {0, nullptr},
};

static PyType_Spec compositor_buffer_spec = {
    /*name*/ "bpy_compositor.Buffer",
    /*basicsize*/ sizeof(BPyCompositorBuffer),
    /*itemsize*/ 0,
    /*flags*/ Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    /*slots*/ compositor_buffer_slots,
};

static PyObject *compositor_buffer_create(const SocketValue &value)
{
  BPyCompositorBuffer *self = reinterpret_cast<BPyCompositorBuffer *>(
      PyType_GenericAlloc(reinterpret_cast<PyTypeObject *>(g_buffer_type), 0));
  if (!self) {
    return nullptr;
  }

  switch (value.type) {
    case compositor::ResultType::Int:
    case compositor::ResultType::Int2:
      self->format[0] = 'i';
      self->itemsize = sizeof(int32_t);
      break;
    case compositor::ResultType::Bool:
      self->format[0] = '?';
      self->itemsize = sizeof(bool);
      break;
    default:
      self->format[0] = 'f';
      self->itemsize = sizeof(float);
      break;
  }
  self->format[1] = '\0';

  self->data = value.data;
  self->readonly = !value.writable;

  /* Row 0 is the bottom row, matching the layout of the compositor. */
  self->shape[0] = value.size.y;
  self->shape[1] = value.size.x;
  if (value.channels == 1) {
    self->ndim = 2;
    self->strides[1] = self->itemsize;
    self->strides[0] = self->strides[1] * self->shape[1];
  }
  else {
    self->ndim = 3;
    self->shape[2] = value.channels;
    self->strides[2] = self->itemsize;
    self->strides[1] = self->strides[2] * self->shape[2];
    self->strides[0] = self->strides[1] * self->shape[1];
  }

  self->sharing = value.sharing.get();
  if (self->sharing) {
    self->sharing->add_user();
  }

  return reinterpret_cast<PyObject *>(self);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Evaluation
 * \{ */

static const char *evaluate_method_name(const EvalMode mode)
{
  return mode == EvalMode::CPU ? "evaluate_cpu" : "evaluate_gpu";
}

static bool compositor_has_method(const bke::bNodeType &ntype, const EvalMode mode)
{
  /* Only Python registered nodes have an owning Python class. */
  if (ntype.rna_ext.data == nullptr) {
    return false;
  }

  PyGILState_STATE gilstate = PyGILState_Ensure();
  PyObject *cls = static_cast<PyObject *>(ntype.rna_ext.data);
  const bool has_method = PyObject_HasAttrString(cls, evaluate_method_name(mode)) != 0;
  PyGILState_Release(gilstate);

  return has_method;
}

static PyObject *tuple_from_floats(const float *values, const int length)
{
  PyObject *tuple = PyTuple_New(length);
  for (int i = 0; i < length; i++) {
    PyTuple_SET_ITEM(tuple, i, PyFloat_FromDouble(values[i]));
  }
  return tuple;
}

static PyObject *tuple_from_ints(const int *values, const int length)
{
  PyObject *tuple = PyTuple_New(length);
  for (int i = 0; i < length; i++) {
    PyTuple_SET_ITEM(tuple, i, PyLong_FromLong(values[i]));
  }
  return tuple;
}

static PyObject *single_value_to_py(const SocketValue &value)
{
  switch (value.type) {
    case compositor::ResultType::Float:
      return PyFloat_FromDouble(value.single_float[0]);
    case compositor::ResultType::Float2:
      return tuple_from_floats(&value.single_float[0], 2);
    case compositor::ResultType::Float3:
      return tuple_from_floats(&value.single_float[0], 3);
    case compositor::ResultType::Float4:
    case compositor::ResultType::Color:
      return tuple_from_floats(&value.single_float[0], 4);
    case compositor::ResultType::Int:
    case compositor::ResultType::Menu:
      return PyLong_FromLong(value.single_int[0]);
    case compositor::ResultType::Int2:
      return tuple_from_ints(&value.single_int[0], 2);
    case compositor::ResultType::Bool:
      return PyBool_FromLong(value.single_int[0]);
    default:
      Py_RETURN_NONE;
  }
}

static PyObject *socket_value_to_py(const SocketValue &value)
{
  switch (value.kind) {
    case SocketValue::Kind::None:
      Py_RETURN_NONE;
    case SocketValue::Kind::Single:
      return single_value_to_py(value);
    case SocketValue::Kind::Buffer:
      return compositor_buffer_create(value);
    case SocketValue::Kind::Texture:
      /* Pass a shared reference, which adds a user to the texture. The deallocation of the Python
       * object removes that user, so it never frees a texture that is owned by the compositor,
       * and a texture retained by Python stays valid, though with undefined contents. */
      return BPyGPUTexture_CreatePyObject(value.texture, true);
  }
  Py_RETURN_NONE;
}

static PyObject *socket_values_to_dict(const Span<SocketValue> values)
{
  PyObject *dict = PyDict_New();
  if (!dict) {
    return nullptr;
  }
  for (const SocketValue &value : values) {
    PyObject *py_value = socket_value_to_py(value);
    if (!py_value) {
      Py_DECREF(dict);
      return nullptr;
    }
    const int result = PyDict_SetItemString(dict, value.identifier.c_str(), py_value);
    Py_DECREF(py_value);
    if (result == -1) {
      Py_DECREF(dict);
      return nullptr;
    }
  }
  return dict;
}

/** Returns a one line description of the current exception and prints its traceback. */
static std::string exception_message_and_print()
{
  std::string message = "Python exception";

  PyObject *exception = PyErr_GetRaisedException();
  if (exception) {
    message = Py_TYPE(exception)->tp_name;
    PyObject *exception_str = PyObject_Str(exception);
    const char *exception_utf8 = exception_str ? PyUnicode_AsUTF8(exception_str) : nullptr;
    if (exception_utf8 && exception_utf8[0] != '\0') {
      message += ": ";
      message += exception_utf8;
    }
    PyErr_Clear();
    Py_XDECREF(exception_str);
    /* Restore the exception, since printing the traceback requires it. */
    PyErr_SetRaisedException(exception);
  }

  PyErr_Print();
  return message;
}

static bool compositor_evaluate(const bNode &node,
                                const EvalMode mode,
                                const Span<SocketValue> inputs,
                                const Span<SocketValue> outputs,
                                std::string &r_error)
{
  PyGILState_STATE gilstate = PyGILState_Ensure();

  PyObject *self = nullptr;
  PyObject *py_inputs = nullptr;
  PyObject *py_outputs = nullptr;
  PyObject *py_result = nullptr;

  /* The type of GPU textures is only initialized once the `gpu` module was imported. */
  PyObject *gpu_types = (mode == EvalMode::GPU) ? PyImport_ImportModule("gpu.types") : nullptr;
  if (mode == EvalMode::CPU || gpu_types) {
    Py_XDECREF(gpu_types);

    PointerRNA ptr = RNA_pointer_create_discrete(const_cast<ID *>(&node.owner_tree().id),
                                                 node.typeinfo->rna_ext.srna,
                                                 const_cast<bNode *>(&node));
    self = pyrna_struct_CreatePyObject(&ptr);
    if (self) {
      py_inputs = socket_values_to_dict(inputs);
    }
    if (py_inputs) {
      py_outputs = socket_values_to_dict(outputs);
    }
    if (py_outputs) {
      py_result = PyObject_CallMethod(
          self, evaluate_method_name(mode), "OO", py_inputs, py_outputs);
    }
  }

  const bool success = py_result != nullptr;
  if (success) {
    Py_DECREF(py_result);
  }
  else {
    r_error = exception_message_and_print();
  }

  Py_XDECREF(self);
  Py_XDECREF(py_inputs);
  Py_XDECREF(py_outputs);

  PyGILState_Release(gilstate);

  return success;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Public API
 * \{ */

static const compositor_python::Callbacks g_callbacks = {
    /*has_method*/ compositor_has_method,
    /*evaluate*/ compositor_evaluate,
};

void BPY_compositor_node_callbacks_register()
{
  if (!g_buffer_type) {
    g_buffer_type = PyType_FromSpec(&compositor_buffer_spec);
    if (!g_buffer_type) {
      PyErr_Print();
      return;
    }
  }
  compositor_python::set_callbacks(&g_callbacks);
}

void BPY_compositor_node_callbacks_unregister()
{
  compositor_python::set_callbacks(nullptr);
  Py_CLEAR(g_buffer_type);
}

/** \} */

}  // namespace blender
