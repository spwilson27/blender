/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Support for compositor nodes that are registered from Python (subclasses of
 * `bpy.types.CompositorNode`) and evaluated by Python code.
 *
 * The nodes and compositor modules must not depend on Python. So the evaluation is routed through
 * a small callback table that the Python module registers at startup, see
 * #blender::nodes::compositor_python::set_callbacks.
 *
 * Python API contract
 * -------------------
 *
 * A Python node class may define any of the following methods, both are optional:
 *
 * \code{.py}
 * def evaluate_cpu(self, inputs, outputs): ...
 * def evaluate_gpu(self, inputs, outputs): ...
 * # Either may optionally take a fourth `context` parameter, see below.
 * def evaluate_cpu(self, inputs, outputs, context): ...
 * \endcode
 *
 * - `evaluate_gpu` is used when the compositor is evaluating on the GPU and the class defines it.
 * - Otherwise `evaluate_cpu` is used. When the compositor evaluates on the GPU, the inputs are
 *   downloaded to the CPU and the outputs are uploaded back to the GPU afterwards.
 * - If only `evaluate_gpu` exists and the compositor evaluates on the CPU, the node outputs
 *   default values and an info message is reported.
 * - If neither exists, the node outputs default values.
 *
 * `inputs` and `outputs` are dictionaries keyed by socket identifier. Outputs that are not needed
 * by any other node are omitted.
 *
 * - Unlinked or constant inputs are passed as a Python `float`, `int`, `bool`, or a `tuple` for
 *   vectors and colors.
 * - Image inputs are passed as buffer objects supporting the buffer protocol (CPU, use
 *   `numpy.asarray`) or as `gpu.types.GPUTexture` (GPU).
 * - Outputs are images allocated over the compute domain of the operation, except for the outputs
 *   listed in `single_value_outputs`, see below. They are zero initialized before the method is
 *   called. Writing to them is done through the writable buffer or by binding the texture as an
 *   image.
 * - Buffers are C-contiguous with shape `(height, width, channels)`, or `(height, width)` for a
 *   single channel, with row 0 at the bottom. Float types use format `f`, integer types `i` and
 *   booleans `?`.
 * - Socket types that are not supported are passed as `None`.
 * - The methods run with the GIL held, possibly not on the main thread, so they should be pure.
 * - If a method raises, the traceback is printed, the outputs fall back to defaults and the error
 *   is reported as the info message of the compositor.
 * - Buffers and textures that were retained by Python remain memory safe, but their contents are
 *   undefined once the method returns.
 *
 * Generator domain
 * ----------------
 *
 * The compute domain is the domain of the image input with the highest domain priority. If the
 * node has no image inputs, that is, every input is unlinked or a single value, the compute domain
 * is the compositing domain (the render resolution) instead of a 1x1 domain.
 *
 * Evaluation context
 * ------------------
 *
 * If the 4th positional parameter of `evaluate_cpu` / `evaluate_gpu` (counting `self`) is named
 * `context`, has no default value (is required), or the function takes `*args`, a fourth argument
 * is passed, a `types.SimpleNamespace` with the attributes:
 *
 * - `frame`: float, the scene frame including the subframe.
 * - `fps`: float, `frs_sec / frs_sec_base` of the scene.
 * - `time`: float, `frame / fps`, in seconds (not offset by the start frame).
 * - `size`: tuple `(width, height)` of the compute domain in pixels.
 * - `use_gpu`: bool, true if the compositor evaluates on the GPU (regardless of which method is
 *   called).
 * - `kind`: str, the kind of evaluation, derived from the compositor context:
 *   `'RENDER'` (the render pipeline: F12, command line render, `render.render()`), `'BACKDROP'`
 *   (the interactive compositor job of the node editor backdrop), `'VIEWPORT'` (the viewport
 *   compositor draw engine) or `'SEQUENCER'` (the compositor modifier of a strip). Streams of
 *   different kinds are interleaved, so stateful nodes should key their state by it.
 * - `is_animation_playing`: bool, true if the animation is playing in the UI. For `'BACKDROP'` it
 *   is the state when the job was scheduled (playback without scrubbing), for `'VIEWPORT'` it is
 *   the state at draw time, for `'RENDER'` and `'SEQUENCER'` it is always false.
 * - `frame_start`, `frame_end`: int, the render frame range (`scene.frame_start` / `frame_end`)
 *   of the scene the compositor context evaluates.
 * - `report(message, level='INFO')`: a function to report a non-fatal message while still
 *   producing output. `level` is `'INFO'` or `'WARNING'` (anything else raises `ValueError`).
 *   After the evaluation the messages are shown as node warnings (Info/Warning) on the node in
 *   the node editor, when the evaluation has a node evaluation log (the node editor backdrop and
 *   render), and the most severe, latest message is also set as the info message of the
 *   compositor (prefixed with the node name). At most #MAX_REPORT_MESSAGES are kept per
 *   evaluation. The function is only valid during the call of the method: calling it afterwards
 *   (for example from a retained reference) raises `RuntimeError`.
 *
 * Methods with three parameters are called exactly as before, and so are methods whose 4th
 * parameter has a default value and another name (for example `_orig=_orig`), which keeps its
 * default. The parameters are read from `__code__` and `__defaults__` of the function, bound
 * methods are unwrapped.
 *
 * Single value outputs
 * --------------------
 *
 * The class attribute `single_value_outputs` can be an iterable of output socket identifiers.
 * Those outputs are single values instead of images (and have no domain). In both the CPU and GPU
 * evaluation, Python receives a writable `bpy_compositor.Buffer` of shape `(channels,)` for them
 * (never a texture), zero initialized, with format `f` for float and color types, `i` for integer
 * types and `?` for booleans. When the method returns successfully, the values are stored as the
 * single value of the output (Float, Float2, Float3, Float4, Color, Int, Int2 and Bool). If the
 * method raises, the outputs have the default value. Unneeded outputs are omitted as usual.
 */

#pragma once

#include <cstdint>
#include <string>

#include "BLI_implicit_sharing_ptr.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "BKE_node.hh"

namespace blender {
struct bNode;
}

namespace blender::gpu {
class Texture;
}

namespace blender::compositor {
enum class ResultType : uint8_t;
}

namespace blender::nodes::compositor_python {

enum class EvalMode {
  CPU,
  GPU,
};

/** Information about the evaluation, passed as the `context` argument to Python. */
struct EvalInfo {
  /* The scene frame including the subframe. */
  float frame = 0.0f;
  /* Frames per second of the scene. */
  float fps = 24.0f;
  /* Time in seconds, `frame / fps`. */
  float time = 0.0f;
  /* Size of the compute domain. */
  int2 size = int2(0);
  /* Whether the compositor evaluates on the GPU. */
  bool use_gpu = false;
  /* The kind of evaluation: "RENDER", "BACKDROP", "VIEWPORT" or "SEQUENCER". Static string. */
  const char *kind = "RENDER";
  /* Whether the animation is playing in the UI. */
  bool is_animation_playing = false;
  /* The render frame range of the evaluated scene. */
  int frame_start = 1;
  int frame_end = 250;
};

/** The maximum number of messages that are kept per evaluation of a node. */
constexpr int MAX_REPORT_MESSAGES = 64;

/** A message reported by Python through `context.report`. */
struct ReportMessage {
  enum class Level {
    Info,
    Warning,
  };
  Level level = Level::Info;
  std::string text;
};

/** A value passed to, or retrieved from, the Python evaluation function of a node. */
struct SocketValue {
  enum class Kind {
    /* Unsupported, passed as `None` to Python. */
    None,
    /* A single value, stored in #single_float or #single_int depending on the type. */
    Single,
    /* A CPU buffer of pixels. */
    Buffer,
    /* A GPU texture. */
    Texture,
  };

  /* The identifier of the socket, which is the key in the dictionary passed to Python. */
  StringRefNull identifier;
  compositor::ResultType type = compositor::ResultType(0);
  Kind kind = Kind::None;

  /* Kind::Single. Unused channels are zero. */
  float4 single_float = float4(0.0f);
  /* Used for int, int2, bool and menu types. */
  int2 single_int = int2(0);

  /* Kind::Buffer. */
  void *data = nullptr;
  /* The buffer holds a single value, so it has the one dimensional shape `(channels,)` and `size`
   * is ignored. */
  bool is_single_value = false;
  int2 size = int2(0);
  int channels = 0;
  bool writable = false;
  /* Keeps `data` alive if it is retained by Python. Can be null for data that outlives the
   * evaluation. */
  ImplicitSharingPtr<> sharing;

  /* Kind::Texture. */
  gpu::Texture *texture = nullptr;
};

struct Callbacks {
  /** Check if the Python class of the given node type defines the method for the given mode. */
  bool (*has_method)(const bke::bNodeType &ntype, EvalMode mode);
  /**
   * Check if the Python class of the given node type lists the given output identifier in its
   * `single_value_outputs` attribute.
   */
  bool (*is_single_value_output)(const bke::bNodeType &ntype, StringRefNull identifier);
  /**
   * Call the method of the given mode. Returns false and fills `r_error` if Python raised an
   * exception. Messages reported through `context.report` are appended to `r_messages`, also on
   * failure. Both spans contain one entry per available input/output socket respectively.
   */
  bool (*evaluate)(const bNode &node,
                   EvalMode mode,
                   Span<SocketValue> inputs,
                   Span<SocketValue> outputs,
                   const EvalInfo &info,
                   Vector<ReportMessage> &r_messages,
                   std::string &r_error);
};

/** Set the callbacks that evaluate Python nodes. Pass nullptr to clear them. */
void set_callbacks(const Callbacks *callbacks);
const Callbacks *get_callbacks();

/** Fill in the compositor specific fields of a Python registered compositor node type. */
void node_type_init(bke::bNodeType &ntype);

}  // namespace blender::nodes::compositor_python
