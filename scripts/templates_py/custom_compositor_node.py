# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Template for a compositor node evaluated in Python.
#
# Both evaluation methods are optional:
# - `evaluate_cpu` receives numpy-compatible buffers, used for CPU compositing
#   (and for GPU compositing when `evaluate_gpu` is missing).
# - `evaluate_gpu` receives gpu.types.GPUTexture objects and dispatches a
#   compute shader, used for GPU compositing.
#
# Inputs and outputs are dicts keyed by socket identifier. Unlinked inputs
# (single values) are passed as Python floats / tuples instead of images.
# Both methods must be pure: do not touch bpy.context, operators or write RNA.
import bpy
import gpu
import numpy as np

_shaders = {}  # Compiled compute shaders, cached per output texture format.


def get_shader(fmt):
    if fmt in _shaders:
        return _shaders[fmt]
    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "src")
    info.image(0, fmt, 'FLOAT_2D', "dst", qualifiers={'WRITE'})
    info.push_constant('FLOAT', "factor")
    info.local_group_size(16, 16)
    info.compute_source('''
void main()
{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(p, imageSize(dst)))) {
    return;
  }
  vec4 c = texelFetch(src, p, 0);
  vec3 inverted = mix(c.rgb, c.a - c.rgb, factor);
  imageStore(dst, p, vec4(inverted, c.a));
}''')
    _shaders[fmt] = gpu.shader.create_from_info(info)
    return _shaders[fmt]


class CompositorNodeInvertPython(bpy.types.CompositorNode):
    '''Invert the colors of an image'''
    bl_idname = "CompositorNodeInvertPython"
    bl_label = "Invert (Python)"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CompositorNodeTree'

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Image")
        factor = self.inputs.new('NodeSocketFloat', "Factor")
        factor.default_value = 1.0
        self.outputs.new('NodeSocketColor', "Image")

    def evaluate_cpu(self, inputs, outputs):
        image = inputs["Image"]
        factor = inputs["Factor"]
        out = np.asarray(outputs["Image"])  # (H, W, 4) float32, row 0 is the bottom row.
        if isinstance(image, tuple):
            # Single value (unlinked socket): the output is a 1x1 image.
            image = np.array(image, np.float32).reshape(1, 1, 4)
        else:
            image = np.asarray(image)  # Read-only view of the compositor's buffer.
        # A linked Factor arrives as a buffer; use its mean for simplicity.
        factor = float(np.mean(factor))
        alpha = image[..., 3:4]
        out[..., :3] = image[..., :3] + factor * (alpha - 2.0 * image[..., :3])
        out[..., 3:4] = alpha

    def evaluate_gpu(self, inputs, outputs):
        image = inputs["Image"]
        dst = outputs["Image"]
        factor = inputs["Factor"]
        if isinstance(image, tuple):
            dst.clear(format='FLOAT', value=image)
            return
        shader = get_shader(dst.format)
        shader.uniform_sampler("src", image)
        shader.image("dst", dst)
        shader.uniform_float("factor", float(factor) if isinstance(factor, float) else 1.0)
        gpu.compute.dispatch(shader, (dst.width + 15) // 16, (dst.height + 15) // 16, 1)


def register():
    bpy.utils.register_class(CompositorNodeInvertPython)


def unregister():
    bpy.utils.unregister_class(CompositorNodeInvertPython)
    _shaders.clear()


if __name__ == "__main__":
    register()
