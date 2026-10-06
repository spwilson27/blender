/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup cmpnodes
 *
 * Compositor operation for nodes that are registered from Python and evaluated by Python code.
 * See NOD_composite_python.hh for a description of the Python API contract.
 */

#include <cstring>
#include <string>

#include "BLI_implicit_sharing.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_vector.hh"

#include "MEM_guardedalloc.h"

#include "DNA_scene_types.h"

#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"

#include "COM_node_operation.hh"
#include "COM_result.hh"
#include "COM_utilities.hh"

#include "NOD_composite_python.hh"
#include "NOD_eval_log.hh"
#include "NOD_menu_value.hh"

namespace blender::nodes::compositor_python {

static const Callbacks *g_callbacks = nullptr;

void set_callbacks(const Callbacks *callbacks)
{
  g_callbacks = callbacks;
}

const Callbacks *get_callbacks()
{
  return g_callbacks;
}

}  // namespace blender::nodes::compositor_python

namespace blender::nodes::node_composite_python_cc {

using namespace blender::compositor;
using compositor_python::Callbacks;
using compositor_python::EvalInfo;
using compositor_python::EvalMode;
using compositor_python::SocketValue;

/* Returns true if images of the given type can be handed to Python as buffers or textures. */
static bool is_supported_image_type(const ResultType type)
{
  switch (type) {
    case ResultType::Float:
    case ResultType::Float2:
    case ResultType::Float3:
    case ResultType::Float4:
    case ResultType::Color:
    case ResultType::Int:
    case ResultType::Int2:
    case ResultType::Bool:
      return true;
    default:
      return false;
  }
}

/* Fill the given socket value from the single value of the given result, or set its kind to None
 * if the type of the result is not supported. */
static void set_single_value(SocketValue &value, const Result &result)
{
  value.kind = SocketValue::Kind::Single;
  switch (result.type()) {
    case ResultType::Float:
      value.single_float = float4(result.get_single_value<float>(), 0.0f, 0.0f, 0.0f);
      break;
    case ResultType::Float2: {
      const float2 float2_value = result.get_single_value<float2>();
      value.single_float = float4(float2_value.x, float2_value.y, 0.0f, 0.0f);
      break;
    }
    case ResultType::Float3:
      value.single_float = float4(result.get_single_value<float3>(), 0.0f);
      break;
    case ResultType::Float4:
      value.single_float = result.get_single_value<float4>();
      break;
    case ResultType::Color:
      value.single_float = float4(result.get_single_value<Color>());
      break;
    case ResultType::Int:
      value.single_int = int2(result.get_single_value<int32_t>(), 0);
      break;
    case ResultType::Int2:
      value.single_int = result.get_single_value<int2>();
      break;
    case ResultType::Bool:
      value.single_int = int2(result.get_single_value<bool>() ? 1 : 0, 0);
      break;
    case ResultType::Menu:
      value.single_int = int2(result.get_single_value<MenuValue>().value, 0);
      break;
    default:
      value.kind = SocketValue::Kind::None;
      break;
  }
}

/* Fill the given socket value to reference the CPU buffer of the given read only result. */
static void set_input_buffer(SocketValue &value, const Result &cpu_result)
{
  const GSpan data = cpu_result.cpu_data();
  value.kind = SocketValue::Kind::Buffer;
  value.size = cpu_result.domain().data_size;
  value.channels = int(cpu_result.channels_count());
  value.writable = false;

  if (cpu_result.sharing_info()) {
    value.data = const_cast<void *>(data.data());
    value.sharing = cpu_result.sharing_info();
    return;
  }

  /* The data is external, so it has no sharing info that Python can use to keep it alive. Python
   * is allowed to retain the buffer, so give it an owned copy to make sure it never dangles. */
  const int64_t size_in_bytes = data.size_in_bytes();
  void *copy = MEM_new_uninitialized(size_in_bytes, __func__);
  memcpy(copy, data.data(), size_in_bytes);
  value.data = copy;
  value.sharing = ImplicitSharingPtr<>(implicit_sharing::info_for_mem_free(copy));
}

/* The number of channels of a single value of a supported type. */
static int single_value_channels(const ResultType type)
{
  switch (type) {
    case ResultType::Float2:
    case ResultType::Int2:
      return 2;
    case ResultType::Float3:
      return 3;
    case ResultType::Float4:
    case ResultType::Color:
      return 4;
    default:
      return 1;
  }
}

/* A single value output that Python writes to, and the output that the value is stored in after
 * the evaluation. */
struct SingleValueOutput {
  Result *output;
  /* The zero initialized storage of the value, kept alive in case Python retains the buffer. */
  ImplicitSharingPtr<> storage;
  void *data;
};

/* Create the writable one dimensional buffer that Python writes a single value output to. */
static SingleValueOutput create_single_value_output(SocketValue &value, Result &output)
{
  /* Large enough for four values of 32 bits. */
  constexpr int64_t storage_size = 4 * sizeof(float);
  void *data = MEM_new_uninitialized(storage_size, __func__);
  memset(data, 0, storage_size);

  value.kind = SocketValue::Kind::Buffer;
  value.is_single_value = true;
  value.data = data;
  value.channels = single_value_channels(output.type());
  value.writable = true;
  value.sharing = ImplicitSharingPtr<>(implicit_sharing::info_for_mem_free(data));
  return {&output, value.sharing, data};
}

/* Allocate the output as a single value that is set to the value written by Python. */
static void store_single_value_output(const SingleValueOutput &single_output)
{
  Result &output = *single_output.output;
  const float *floats = static_cast<const float *>(single_output.data);
  const int *ints = static_cast<const int *>(single_output.data);
  output.allocate_single_value();
  switch (output.type()) {
    case ResultType::Float:
      output.set_single_value(floats[0]);
      break;
    case ResultType::Float2:
      output.set_single_value(float2(floats[0], floats[1]));
      break;
    case ResultType::Float3:
      output.set_single_value(float3(floats[0], floats[1], floats[2]));
      break;
    case ResultType::Float4:
      output.set_single_value(float4(floats[0], floats[1], floats[2], floats[3]));
      break;
    case ResultType::Color:
      output.set_single_value(Color(floats[0], floats[1], floats[2], floats[3]));
      break;
    case ResultType::Int:
      output.set_single_value(int32_t(ints[0]));
      break;
    case ResultType::Int2:
      output.set_single_value(int2(ints[0], ints[1]));
      break;
    case ResultType::Bool:
      output.set_single_value(*static_cast<const bool *>(single_output.data));
      break;
    default:
      BLI_assert_unreachable();
      break;
  }
}

/* A temporary CPU result that Python writes to, and the output that its data is passed to after
 * the evaluation. */
struct TemporaryOutput {
  Result *output;
  Result temporary;
};

class PythonNodeOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    const Callbacks *callbacks = compositor_python::get_callbacks();
    if (!callbacks) {
      this->allocate_default_remaining_outputs();
      return;
    }

    const bke::bNodeType &ntype = *this->node().typeinfo;
    const bool has_gpu_method = callbacks->has_method(ntype, EvalMode::GPU);
    const bool has_cpu_method = callbacks->has_method(ntype, EvalMode::CPU);

    if (this->context().use_gpu() && has_gpu_method) {
      this->execute_gpu(*callbacks);
    }
    else if (has_cpu_method) {
      this->execute_cpu(*callbacks);
    }
    else {
      if (has_gpu_method) {
        this->report_message("requires GPU compositing");
      }
      this->allocate_default_remaining_outputs();
    }
  }

 public:
  Domain compute_domain() override
  {
    /* The base implementation returns the identity domain if there is no image input to take the
     * domain from. Python nodes without such inputs are generators, so they use the compositing
     * domain instead. Mirror the criteria of the base implementation to detect that case. */
    bool has_domain_input = false;
    for (const bNodeSocket *socket : this->node().input_sockets()) {
      if (!is_socket_available(socket)) {
        continue;
      }
      const Result &input = this->get_input(socket->identifier);
      const InputDescriptor &descriptor = this->get_input_descriptor(socket->identifier);
      if (input.is_single_value() || descriptor.expects_single_value) {
        continue;
      }
      if (descriptor.realization_mode != InputRealizationMode::OperationDomain) {
        continue;
      }
      has_domain_input = true;
      break;
    }

    if (!has_domain_input) {
      return this->context().get_compositing_domain();
    }
    return NodeOperation::compute_domain();
  }

 private:
  EvalInfo get_eval_info(const Domain &domain) const
  {
    const RenderData &render_data = this->context().get_render_data();
    EvalInfo info;
    info.frame = float(render_data.cfra) + render_data.subframe;
    info.fps = float(render_data.frs_sec) / render_data.frs_sec_base;
    info.time = info.fps != 0.0f ? info.frame / info.fps : 0.0f;
    info.size = domain.data_size;
    info.use_gpu = this->context().use_gpu();
    switch (this->context().get_evaluation_kind()) {
      case compositor::EvaluationKind::Render:
        info.kind = "RENDER";
        break;
      case compositor::EvaluationKind::Backdrop:
        info.kind = "BACKDROP";
        break;
      case compositor::EvaluationKind::Viewport:
        info.kind = "VIEWPORT";
        break;
      case compositor::EvaluationKind::Sequencer:
        info.kind = "SEQUENCER";
        break;
    }
    info.is_animation_playing = this->context().is_animation_playing();
    info.frame_start = this->context().get_scene().r.sfra;
    info.frame_end = this->context().get_scene().r.efra;
    return info;
  }

  void report_message(const StringRef message) const
  {
    this->context().set_info_message(std::string(this->node().name) + ": " + std::string(message));
  }

  /* Show the messages reported by Python as node warnings, and as the info message of the
   * context. Errors are reported separately, as the info message. */
  void forward_messages(const Span<compositor_python::ReportMessage> messages) const
  {
    using compositor_python::ReportMessage;
    if (messages.is_empty()) {
      return;
    }

    if (nodes::eval_log::NodesEvalLog *log = this->context().nodes_evaluation_log()) {
      nodes::eval_log::NodeTreeLogger &tree_logger = log->get_local_tree_logger(
          this->get_compute_context());
      for (const ReportMessage &message : messages) {
        const NodeWarningType type = message.level == ReportMessage::Level::Warning ?
                                         NodeWarningType::Warning :
                                         NodeWarningType::Info;
        tree_logger.node_warnings.append(*tree_logger.allocator,
                                         {this->node().identifier, {type, message.text}});
      }
    }

    /* The most severe message wins, the latest one among those of the same severity. */
    const ReportMessage *shown = nullptr;
    for (const ReportMessage &message : messages) {
      if (!shown || message.level >= shown->level) {
        shown = &message;
      }
    }
    this->report_message(shown->text);
  }

  void execute_cpu(const Callbacks &callbacks)
  {
    const bool use_gpu = this->context().use_gpu();
    const Domain domain = this->compute_domain();

    /* Inputs that were downloaded from the GPU and need to be released after the evaluation. */
    Vector<Result> temporary_inputs;
    Vector<SocketValue> inputs;
    for (const bNodeSocket *socket : this->node().input_sockets()) {
      if (!is_socket_available(socket)) {
        continue;
      }

      const Result &input = this->get_input(socket->identifier);
      SocketValue value;
      value.identifier = socket->identifier;
      value.type = input.type();
      if (input.is_single_value()) {
        set_single_value(value, input);
      }
      else if (is_supported_image_type(input.type())) {
        if (use_gpu) {
          temporary_inputs.append(input.download_to_cpu());
          set_input_buffer(value, temporary_inputs.last());
        }
        else {
          set_input_buffer(value, input);
        }
      }
      inputs.append(std::move(value));
    }

    Vector<TemporaryOutput> temporary_outputs;
    Vector<SingleValueOutput> single_value_outputs;
    Vector<SocketValue> outputs;
    for (const bNodeSocket *socket : this->node().output_sockets()) {
      if (!is_socket_available(socket)) {
        continue;
      }

      Result &output = this->get_result(socket->identifier);
      if (!output.should_compute()) {
        continue;
      }

      SocketValue value;
      value.identifier = socket->identifier;
      value.type = output.type();
      if (!is_supported_image_type(output.type())) {
        output.allocate_invalid();
        outputs.append(std::move(value));
        continue;
      }

      if (callbacks.is_single_value_output(*this->node().typeinfo, socket->identifier)) {
        single_value_outputs.append(create_single_value_output(value, output));
        outputs.append(std::move(value));
        continue;
      }

      /* The temporary result is always allocated on the CPU, even if the context uses the GPU. */
      Result temporary = this->context().create_result(output.type(), output.precision());
      temporary.allocate_texture(domain, false, ResultStorageType::CPU);

      /* Make sure an output that Python does not write to is zero instead of undefined. */
      const GMutableSpan data = temporary.cpu_data_for_write();
      memset(data.data(), 0, data.size_in_bytes());

      value.kind = SocketValue::Kind::Buffer;
      value.data = data.data();
      value.size = temporary.domain().data_size;
      value.channels = int(temporary.channels_count());
      value.writable = true;
      value.sharing = temporary.sharing_info();
      outputs.append(std::move(value));
      temporary_outputs.append({&output, std::move(temporary)});
    }

    std::string error;
    Vector<compositor_python::ReportMessage> messages;
    const bool success = callbacks.evaluate(this->node(),
                                            EvalMode::CPU,
                                            inputs,
                                            outputs,
                                            this->get_eval_info(domain),
                                            messages,
                                            error);
    this->forward_messages(messages);

    if (success) {
      for (const SingleValueOutput &single_output : single_value_outputs) {
        store_single_value_output(single_output);
      }
    }

    for (TemporaryOutput &temporary_output : temporary_outputs) {
      if (success) {
        if (use_gpu) {
          Result gpu_result = temporary_output.temporary.upload_to_gpu(true);
          temporary_output.output->share_data(gpu_result);
          gpu_result.release();
        }
        else {
          temporary_output.output->share_data(temporary_output.temporary);
        }
      }
      temporary_output.temporary.release();
    }

    for (Result &temporary_input : temporary_inputs) {
      temporary_input.release();
    }

    if (!success) {
      this->report_message(error);
    }

    /* Outputs that were not written in case of an error, or that are not supported. */
    this->allocate_default_remaining_outputs();
  }

  void execute_gpu(const Callbacks &callbacks)
  {
    const Domain domain = this->compute_domain();

    Vector<SocketValue> inputs;
    for (const bNodeSocket *socket : this->node().input_sockets()) {
      if (!is_socket_available(socket)) {
        continue;
      }

      const Result &input = this->get_input(socket->identifier);
      SocketValue value;
      value.identifier = socket->identifier;
      value.type = input.type();
      if (input.is_single_value()) {
        set_single_value(value, input);
      }
      else if (is_supported_image_type(input.type())) {
        value.kind = SocketValue::Kind::Texture;
        value.texture = input.gpu_texture();
      }
      inputs.append(std::move(value));
    }

    Vector<Result *> allocated_outputs;
    Vector<SingleValueOutput> single_value_outputs;
    Vector<SocketValue> outputs;
    for (const bNodeSocket *socket : this->node().output_sockets()) {
      if (!is_socket_available(socket)) {
        continue;
      }

      Result &output = this->get_result(socket->identifier);
      if (!output.should_compute()) {
        continue;
      }

      SocketValue value;
      value.identifier = socket->identifier;
      value.type = output.type();
      if (!is_supported_image_type(output.type())) {
        output.allocate_invalid();
        outputs.append(std::move(value));
        continue;
      }

      if (callbacks.is_single_value_output(*this->node().typeinfo, socket->identifier)) {
        single_value_outputs.append(create_single_value_output(value, output));
        outputs.append(std::move(value));
        continue;
      }

      output.allocate_texture(domain);
      this->clear_output(output);
      allocated_outputs.append(&output);

      value.kind = SocketValue::Kind::Texture;
      value.texture = output.gpu_texture();
      outputs.append(std::move(value));
    }

    std::string error;
    Vector<compositor_python::ReportMessage> messages;
    const bool success = callbacks.evaluate(this->node(),
                                            EvalMode::GPU,
                                            inputs,
                                            outputs,
                                            this->get_eval_info(domain),
                                            messages,
                                            error);
    this->forward_messages(messages);

    /* Python is expected to leave no shader bound, but make sure that is the case, and that the
     * writes to the output textures are visible to the operations that consume them. */
    GPU_shader_unbind();
    GPU_texture_unbind_all();
    GPU_texture_image_unbind_all();
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH | GPU_BARRIER_SHADER_IMAGE_ACCESS |
                       GPU_BARRIER_TEXTURE_UPDATE);

    if (success) {
      for (const SingleValueOutput &single_output : single_value_outputs) {
        store_single_value_output(single_output);
      }
    }
    else {
      /* Python might have partially written the outputs, so go back to the default. */
      for (Result *output : allocated_outputs) {
        this->clear_output(*output);
      }
      this->report_message(error);
    }

    this->allocate_default_remaining_outputs();
  }

  void clear_output(Result &output)
  {
    /* Large enough for a pixel of any of the supported types. */
    const float4 zero_value = float4(0.0f);
    GPU_texture_clear(output.gpu_texture(), output.get_gpu_data_format(), &zero_value);
  }
};

static NodeOperation *get_compositor_operation(Context &context, const bNode &node)
{
  return new PythonNodeOperation(context, node);
}

}  // namespace blender::nodes::node_composite_python_cc

namespace blender::nodes::compositor_python {

void node_type_init(bke::bNodeType &ntype)
{
  ntype.get_compositor_operation = node_composite_python_cc::get_compositor_operation;
}

}  // namespace blender::nodes::compositor_python
