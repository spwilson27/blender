/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup pythonintern
 *
 * Evaluation of Python registered compositor nodes, see NOD_composite_python.hh.
 */

#pragma once

namespace blender {

/**
 * Register the callbacks that let the compositor call the `evaluate_cpu` and `evaluate_gpu`
 * methods of Python registered compositor nodes. Called with the GIL held.
 */
void BPY_compositor_node_callbacks_register();

/**
 * Unregister the callbacks and free the Python types they use. Called with the GIL held.
 */
void BPY_compositor_node_callbacks_unregister();

}  // namespace blender
