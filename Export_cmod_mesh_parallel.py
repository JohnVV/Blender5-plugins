"""
CMOD Mesh Exporter for Blender 5  —  Multi-Core Refactor
Original script copyright (C) Stephen Popovich 2009 - for blender 2.48
Updated for Blender 5 John Van Vliet



# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# ***** END GPL LICENCE BLOCK *****

bl_info = {
    "name": "CMOD Mesh Exporter (parallel)",
    "author": "John Van Vliet (Blender 5.0.1), multi-core refactor 2025",
    "version": (1, 0, 0),
    "blender": (5, 0, 1),
    "location": "File > Export > Celestia Mesh (.cmod)",
    "description": "Export selected meshes to Celestia .cmod ASCII format "
                   "(multi-core, numpy-accelerated, "
                   "Texture/specular/emissive/normal maps + General options)",
    "category": "Export",
}

import bpy
import bmesh
import os
import io
import math
import multiprocessing
import concurrent.futures
from bpy_extras.io_utils import ExportHelper, axis_conversion
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    EnumProperty,
    IntProperty,
)
from bpy.types import Operator
import mathutils

# ---------------------------------------------------------------------------
# Axis-forward / axis-up helpers
# ---------------------------------------------------------------------------

AXIS_ITEMS = [
    ('X',  "X",  ""),
    ('Y',  "Y",  ""),
    ('Z',  "Z",  ""),
    ('-X', "-X", ""),
    ('-Y', "-Y", ""),
    ('-Z', "-Z", ""),
]

# Minimum face count per object before bothering with a worker process.
# Below this threshold process-pool overhead outweighs the speed gain.
_PARALLEL_FACE_THRESHOLD = 2_000


# ---------------------------------------------------------------------------
# Helpers – material / texture  (unchanged, main-thread only)
# ---------------------------------------------------------------------------

def _linked_tex_image(socket):
    """Return the image node linked to *socket*, or None."""
    if socket and socket.is_linked:
        from_node = socket.links[0].from_node
        if from_node.type == 'NORMAL_MAP':
            color_socket = from_node.inputs.get("Color")
            if color_socket and color_socket.is_linked:
                from_node = color_socket.links[0].from_node
        if from_node.type == 'TEX_IMAGE' and from_node.image:
            return from_node.image
    return None


def _resolve_tex_path(image, cmod_filepath, path_mode):
    """
    Return the texture path string according to *path_mode*.

    BASENAME  – just the filename  (recommended for Celestia)
    RELATIVE  – path relative to the exported .cmod file
    ABSOLUTE  – full absolute path
    """
    if image is None:
        return None

    raw = bpy.path.abspath(image.filepath)
    if not raw:
        raw = image.name

    if path_mode == 'BASENAME':
        return os.path.basename(raw) if raw else image.name
    elif path_mode == 'RELATIVE':
        cmod_dir = os.path.dirname(os.path.abspath(cmod_filepath))
        try:
            rel = os.path.relpath(raw, cmod_dir)
        except ValueError:
            rel = raw
        return rel.replace('\\', '/')
    else:  # ABSOLUTE
        return os.path.abspath(raw).replace('\\', '/')


def get_material_info(mat, has_uv, cmod_filepath='', path_mode='BASENAME'):
    """
    Extract material data from a Blender 5 material node tree.

    Returned dict keys
    ------------------
    dif_r/g/b       – diffuse colour
    spec_r/g/b      – specular colour
    spec_pwr        – specular power  (1–128)
    opacity         – alpha
    emit            – emission strength (scalar)
    texture         – base-color / diffuse image path  (texture0)
    specular_map    – specular map path                (specularmap)
    emissive_map    – emissive map path                (emissivemap)
    normal_map      – normal map path                  (normalmap)
    """
    info = {
        'dif_r': 1.0, 'dif_g': 1.0, 'dif_b': 1.0,
        'spec_r': 0.5, 'spec_g': 0.5, 'spec_b': 0.5,
        'spec_pwr': 0.0,
        'opacity': 1.0,
        'emit': 0.0,
        'texture':      None,
        'specular_map': None,
        'emissive_map': None,
        'normal_map':   None,
    }

    def _path(image):
        return _resolve_tex_path(image, cmod_filepath, path_mode)

    if mat is None:
        return info

    if mat.use_nodes and mat.node_tree:
        nodes      = mat.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        emission   = nodes.get("Emission")

        if principled:
            base_color    = principled.inputs["Base Color"].default_value
            info['dif_r'] = base_color[0]
            info['dif_g'] = base_color[1]
            info['dif_b'] = base_color[2]
            info['texture'] = _path(_linked_tex_image(principled.inputs.get("Base Color")))

            spec_input = (principled.inputs.get("Specular IOR Level")
                          or principled.inputs.get("Specular"))
            if spec_input:
                s = spec_input.default_value
                info['spec_r'] = s
                info['spec_g'] = s
                info['spec_b'] = s
                info['specular_map'] = _path(_linked_tex_image(spec_input))

            roughness        = principled.inputs["Roughness"].default_value
            info['spec_pwr'] = max(1.0, (1.0 - roughness) * 128.0)

            alpha_input = principled.inputs.get("Alpha")
            if alpha_input:
                info['opacity'] = alpha_input.default_value

            emit_strength = principled.inputs.get("Emission Strength")
            if emit_strength:
                info['emit'] = emit_strength.default_value

            emit_color_socket = (principled.inputs.get("Emission Color")
                                 or principled.inputs.get("Emission"))
            info['emissive_map'] = _path(_linked_tex_image(emit_color_socket))

            normal_socket = principled.inputs.get("Normal")
            info['normal_map'] = _path(_linked_tex_image(normal_socket))

        elif emission:
            color             = emission.inputs["Color"].default_value
            info['dif_r']     = color[0]
            info['dif_g']     = color[1]
            info['dif_b']     = color[2]
            info['emit']      = emission.inputs["Strength"].default_value
            info['emissive_map'] = _path(_linked_tex_image(emission.inputs.get("Color")))

        else:
            for node in nodes:
                if node.type != 'TEX_IMAGE' or not node.image:
                    continue
                label    = node.label.lower()
                name     = node.image.name.lower()
                tag      = label or name
                filename = _path(node.image)
                if 'normal' in tag or 'nrm' in tag or 'nor' in tag:
                    info['normal_map']   = info['normal_map']   or filename
                elif 'spec' in tag:
                    info['specular_map'] = info['specular_map'] or filename
                elif 'emit' in tag or 'emissive' in tag or 'glow' in tag:
                    info['emissive_map'] = info['emissive_map'] or filename
                else:
                    info['texture']      = info['texture']      or filename
    else:
        dc             = mat.diffuse_color
        info['dif_r']  = dc[0]
        info['dif_g']  = dc[1]
        info['dif_b']  = dc[2]
        info['opacity'] = dc[3] if len(dc) > 3 else 1.0

    return info


# ---------------------------------------------------------------------------
# Writer helpers – material block
# ---------------------------------------------------------------------------

def _write_material(buf, minfo):
    """Write a single material block to StringIO *buf*."""
    buf.write('\nmaterial\n')

    if minfo['texture']:
        buf.write('texture0 "%s"\n' % minfo['texture'])
    if minfo['emissive_map']:
        buf.write('emissivemap "%s"\n' % minfo['emissive_map'])
    if minfo['specular_map']:
        buf.write('specularmap "%s"\n' % minfo['specular_map'])
    if minfo['normal_map']:
        buf.write('normalmap "%s"\n' % minfo['normal_map'])

    if minfo['emit'] > 0 and not minfo['emissive_map']:
        buf.write('emissive %f %f %f\n' % (
            minfo['dif_r'] * minfo['emit'],
            minfo['dif_g'] * minfo['emit'],
            minfo['dif_b'] * minfo['emit']))
    else:
        buf.write('diffuse %f %f %f\n' % (
            minfo['dif_r'], minfo['dif_g'], minfo['dif_b']))

    buf.write('specular %f %f %f\n' % (
        minfo['spec_r'], minfo['spec_g'], minfo['spec_b']))

    if minfo['spec_pwr'] > 0:
        buf.write('specpower %f\n' % minfo['spec_pwr'])

    if minfo['opacity'] < 1.0:
        buf.write('opacity %f\n' % minfo['opacity'])

    buf.write('end_material\n')


# ---------------------------------------------------------------------------
# Phase-1 helper  (main thread) – extract raw loop data from bmesh
# ---------------------------------------------------------------------------

def _extract_loop_data(bm, uv_layer):
    """
    Pull all per-loop geometry data out of *bm* into plain Python lists.

    The returned structure is fully picklable so it can be sent to a worker
    process without any bpy dependency.

    Returns
    -------
    face_loops_list : list[list[tuple]]
        Outer list = one entry per face.
        Inner list = one tuple per loop: (x, y, z, nx, ny, nz [, u, v]).
    has_uv : bool
    """
    has_uv = uv_layer is not None
    face_loops_list = []

    for face in bm.faces:
        face_normal = (face.normal.x, face.normal.y, face.normal.z)
        smooth      = face.smooth
        loops_out   = []
        for loop in face.loops:
            co = loop.vert.co
            if smooth:
                n = loop.calc_normal()
                no = (n.x, n.y, n.z)
            else:
                no = face_normal
            if has_uv:
                uv = loop[uv_layer].uv
                loops_out.append((co.x, co.y, co.z,
                                  no[0], no[1], no[2],
                                  uv.x, 1.0 - uv.y))   # flip V for OpenGL
            else:
                loops_out.append((co.x, co.y, co.z,
                                  no[0], no[1], no[2]))
        face_loops_list.append(loops_out)

    return face_loops_list, has_uv


# ---------------------------------------------------------------------------
# Phase-2 workers  (run in worker processes – NO bpy imports allowed here)
# ---------------------------------------------------------------------------

def _build_vertices_numpy(face_loops_list, has_uv):
    """
    Numpy-accelerated vertex deduplication and index building.

    Roughly 5–10× faster than the dict loop for meshes with > ~20 k faces.
    Falls back to the dict path if numpy is not available.

    Parameters
    ----------
    face_loops_list : list[list[tuple]]  (from _extract_loop_data)
    has_uv          : bool

    Returns
    -------
    vertices : list[str]   – one formatted line per unique vertex
    indices  : list[int]   – flat triangle index list
    """
    try:
        import numpy as np
    except ImportError:
        return _build_vertices_dict(face_loops_list, has_uv)

    if not face_loops_list:
        return [], []

    # Flatten all loop tuples into a 2-D float array
    all_loops  = [loop for face in face_loops_list for loop in face]
    loop_sizes = [len(face) for face in face_loops_list]
    arr = np.array(all_loops, dtype=np.float64)

    # Round to 6 d.p. so that near-duplicate vertices merge correctly
    arr_r = np.round(arr, 6)

    # Build a structured dtype with one field per component so np.unique
    # treats each row as a single comparable key.
    ncols  = arr_r.shape[1]
    dtype  = np.dtype([('f%d' % i, np.float64) for i in range(ncols)])
    struct = np.ascontiguousarray(arr_r).view(dtype).ravel()

    unique_struct, inverse = np.unique(struct, return_inverse=True)
    unique_arr = unique_struct.view(np.float64).reshape(-1, ncols)

    # Format vertex strings using vectorised string ops
    fmt_cols = ['%.6f' % v for row in unique_arr for v in row]
    stride   = ncols
    vertices = [
        ' '.join(fmt_cols[i * stride:(i + 1) * stride])
        for i in range(len(unique_arr))
    ]

    # Reconstruct triangle indices from per-face sizes
    indices = []
    pos     = 0
    for size in loop_sizes:
        fi = list(map(int, inverse[pos:pos + size]))
        if size == 3:
            indices.extend(fi)
        elif size > 3:
            for t in range(1, size - 1):
                indices.extend([fi[0], fi[t], fi[t + 1]])
        pos += size

    return vertices, indices


def _build_vertices_dict(face_loops_list, has_uv):
    """
    Pure-Python vertex deduplication and index building.

    This is the original algorithm, refactored to work on plain Python data
    (no bpy/bmesh) so it is safe to run in a worker process.

    Parameters
    ----------
    face_loops_list : list[list[tuple]]  (from _extract_loop_data)
    has_uv          : bool

    Returns
    -------
    vertices : list[str]
    indices  : list[int]
    """
    vertex_map = {}   # rounded-tuple key → int index
    vertices   = []
    indices    = []

    def _fmt(x):
        return '%.6f' % x

    for face_loops in face_loops_list:
        tri_indices = []
        for data in face_loops:
            key = tuple(round(v, 6) for v in data)
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(' '.join(_fmt(v) for v in key))
            tri_indices.append(vertex_map[key])

        if len(tri_indices) == 3:
            indices.extend(tri_indices)
        elif len(tri_indices) > 3:
            for t in range(1, len(tri_indices) - 1):
                indices.extend([tri_indices[0], tri_indices[t], tri_indices[t + 1]])

    return vertices, indices


def _worker_build_mesh(args):
    """
    Top-level worker function submitted to ProcessPoolExecutor.

    Must be defined at module level so it is importable by spawned processes.

    Parameters
    ----------
    args : tuple
        (object_index, face_loops_list, has_uv, use_numpy)

    Returns
    -------
    (object_index, vertices, indices)
    """
    obj_index, face_loops_list, has_uv, use_numpy = args

    if use_numpy:
        vertices, indices = _build_vertices_numpy(face_loops_list, has_uv)
    else:
        vertices, indices = _build_vertices_dict(face_loops_list, has_uv)

    return obj_index, vertices, indices


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def write_cmod(filepath, context, operator):
    """
    Main export function.  Three-phase pipeline:

    Phase 1  – main thread: bpy/bmesh evaluation → raw loop data
    Phase 2  – worker pool: vertex dedup + index build + string format
    Phase 3  – main thread: buffered StringIO assembly → single file write

    Parameters read from *operator*
    --------------------------------
    use_selection       – export only selected objects vs. all visible
    use_visible_only    – skip hidden objects when exporting all
    use_apply_modifiers – apply viewport modifiers before export
    global_scale        – uniform scale factor applied to geometry
    axis_forward        – forward axis remapping  (e.g. '-Z')
    axis_up             – up axis remapping       (e.g. 'Y')
    use_triangulate     – force triangulation
    use_materials       – write material blocks
    use_uvs             – include UV coords
    path_mode           – BASENAME / RELATIVE / ABSOLUTE
    worker_count        – number of worker processes (0 = auto)
    use_numpy           – prefer numpy-accelerated vertex build
    """

    # ---- Gather objects ----
    if operator.use_selection:
        candidates = context.selected_objects
    else:
        candidates = context.scene.objects

    objects = [
        obj for obj in candidates
        if obj.type == 'MESH'
        and (not operator.use_visible_only or obj.visible_get())
    ]

    if not objects:
        return {'CANCELLED'}, "No mesh objects to export."

    # ---- Axis conversion matrix ----
    axis_mat = axis_conversion(
        to_forward=operator.axis_forward,
        to_up=operator.axis_up,
    ).to_4x4()

    scale_mat     = mathutils.Matrix.Scale(operator.global_scale, 4)
    global_matrix = scale_mat @ axis_mat

    depsgraph = context.evaluated_depsgraph_get()

    # ====================================================================
    # PHASE 1 – main thread
    # Extract bpy/bmesh data into plain picklable structures.
    # ====================================================================
    raw_records = []   # list of (minfo, face_loops_list, has_uv)

    for obj in objects:
        obj_eval = obj.evaluated_get(depsgraph)

        if operator.use_apply_modifiers:
            mesh = obj_eval.to_mesh(
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )
        else:
            mesh = obj.to_mesh(
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )

        mesh.calc_loop_triangles()

        bm = bmesh.new()
        bm.from_mesh(mesh)

        if operator.use_triangulate:
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

        bm.transform(global_matrix @ obj.matrix_world)
        bm.normal_update()

        uv_layer = (bm.loops.layers.uv.active
                    if (operator.use_uvs and bm.loops.layers.uv)
                    else None)

        mat   = obj.material_slots[0].material if obj.material_slots else None
        minfo = get_material_info(
            mat,
            uv_layer is not None,
            cmod_filepath=filepath,
            path_mode=operator.path_mode,
        )

        # Extract raw loop data – no bpy objects survive past this point
        face_loops_list, has_uv = _extract_loop_data(bm, uv_layer)
        raw_records.append((minfo, face_loops_list, has_uv))

        bm.free()
        obj_eval.to_mesh_clear()

    if not raw_records:
        return {'CANCELLED'}, "No exportable geometry found."

    # ====================================================================
    # PHASE 2 – parallel worker pool
    # Vertex dedup, index building, and string formatting.
    # ====================================================================

    # Determine how many workers to use.
    cpu_count = multiprocessing.cpu_count()
    if operator.worker_count == 0:
        max_workers = cpu_count
    else:
        max_workers = min(operator.worker_count, cpu_count)

    use_numpy = operator.use_numpy

    # Only spawn worker processes when it is worth the overhead:
    # we need more than one object, and at least one has enough faces.
    big_enough = any(
        len(fl) >= _PARALLEL_FACE_THRESHOLD
        for _, fl, _ in raw_records
    )
    use_parallel = (max_workers > 1
                    and len(raw_records) > 1
                    and big_enough)

    mesh_results = [None] * len(raw_records)   # preserve insertion order

    if use_parallel:
        worker_args = [
            (i, fl, has_uv, use_numpy)
            for i, (_, fl, has_uv) in enumerate(raw_records)
        ]
        try:
            # 'spawn' start method is safest inside Blender (avoids fork issues).
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx) as pool:
                futures = {pool.submit(_worker_build_mesh, arg): arg[0]
                           for arg in worker_args}
                for fut in concurrent.futures.as_completed(futures):
                    obj_index, vertices, indices = fut.result()
                    minfo, _, has_uv = raw_records[obj_index]
                    mesh_results[obj_index] = (minfo, has_uv, vertices, indices)
        except Exception as exc:
            # Graceful fallback: process sequentially if the pool fails.
            print("CMOD exporter: worker pool failed (%s), falling back to "
                  "single-threaded mode." % exc)
            use_parallel = False

    if not use_parallel:
        # Single-threaded path: also used for single-object scenes.
        for i, (minfo, face_loops_list, has_uv) in enumerate(raw_records):
            if use_numpy:
                vertices, indices = _build_vertices_numpy(face_loops_list, has_uv)
            else:
                vertices, indices = _build_vertices_dict(face_loops_list, has_uv)
            mesh_results[i] = (minfo, has_uv, vertices, indices)

    # ====================================================================
    # PHASE 3 – main thread
    # Assemble output with a StringIO buffer, then write once.
    # ====================================================================
    buf = io.StringIO()
    buf.write('#celmodel__ascii\n')

    # --- Material blocks (all written first, in order) ---
    if operator.use_materials:
        for minfo, has_uv, vertices, indices in mesh_results:
            _write_material(buf, minfo)

    # --- Mesh blocks ---
    for mat_index, (minfo, has_uv, vertices, indices) in enumerate(mesh_results):
        buf.write('\nmesh\n')

        buf.write('vertexdesc\n')
        buf.write('position f3\n')
        buf.write('normal f3\n')
        if has_uv:
            buf.write('texcoord0 f2\n')
        buf.write('end_vertexdesc\n\n')

        num_verts = len(vertices)
        buf.write('vertices %i\n' % num_verts)
        # Write all vertex lines in one join to minimise buf.write() calls
        buf.write('\n'.join(vertices))
        buf.write('\n')

        num_indices = len(indices)
        buf.write('\ntrilist %i %i\n' % (
            mat_index if operator.use_materials else 0,
            num_indices,
        ))

        # Write indices 12 per line
        for start in range(0, num_indices, 12):
            buf.write(' '.join(str(i) for i in indices[start:start + 12]))
            buf.write('\n')

        buf.write('\nend_mesh\n')

    # Single write to disk
    content = buf.getvalue()
    buf.close()

    with open(filepath, 'w', encoding='utf-8') as out:
        out.write(content)

    num_tris = sum(len(ind) // 3 for _, _, _, ind in mesh_results)
    parallel_note = ((" (parallel, %d workers)" % max_workers)
                     if use_parallel else " (single-threaded)")
    return {'FINISHED'}, (
        "Export complete%s: %s  (%d mesh(es), %d triangles)"
        % (parallel_note, filepath, len(mesh_results), num_tris)
    )


# ---------------------------------------------------------------------------
# Operator  (with General options panel)
# ---------------------------------------------------------------------------

class ExportCMOD(Operator, ExportHelper):
    """Export selected meshes to Celestia .cmod format (multi-core)"""
    bl_idname  = "export_mesh.cmod"
    bl_label   = "Export CMOD"
    bl_options = {'PRESET'}

    filename_ext = ".cmod"

    filter_glob: StringProperty(
        default="*.cmod",
        options={'HIDDEN'},
        maxlen=255,
    )

    # ------------------------------------------------------------------ #
    #  G E N E R A L   O P T I O N S                                      #
    # ------------------------------------------------------------------ #

    # --- Include ---
    use_selection: BoolProperty(
        name="Selected Objects Only",
        description="Export only the currently selected objects; "
                    "when disabled all visible scene objects are exported",
        default=True,
    )

    use_visible_only: BoolProperty(
        name="Visible Objects Only",
        description="Skip objects that are hidden in the viewport "
                    "(only relevant when 'Selected Objects Only' is off)",
        default=True,
    )

    # --- Transform ---
    global_scale: FloatProperty(
        name="Scale",
        description="Uniform scale applied to all exported geometry",
        min=0.0001, max=10000.0,
        soft_min=0.01, soft_max=1000.0,
        default=1.0,
        step=10,
        precision=4,
    )

    axis_forward: EnumProperty(
        name="Forward",
        description="Which Blender axis maps to the model's forward direction",
        items=AXIS_ITEMS,
        default='X',
    )

    axis_up: EnumProperty(
        name="Up",
        description="Which Blender axis maps to the model's up direction",
        items=AXIS_ITEMS,
        default='Z',
    )

    # --- Geometry ---
    use_apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply viewport modifiers to the mesh before exporting",
        default=True,
    )

    use_triangulate: BoolProperty(
        name="Triangulate Faces",
        description="Force triangulation of all polygons before export "
                    "(required by the .cmod format; disable only if your "
                    "mesh is already fully triangulated)",
        default=True,
    )

    use_smooth_groups: BoolProperty(
        name="Write Smooth Groups",
        description="Preserve Blender's smooth/flat shading as per-loop "
                    "split normals (affects per-vertex normal direction)",
        default=True,
    )

    # --- Material / Texture ---
    use_materials: BoolProperty(
        name="Export Materials",
        description="Write material blocks to the .cmod file",
        default=True,
    )

    use_uvs: BoolProperty(
        name="Export UV Coordinates",
        description="Include UV texture coordinates in the vertex data",
        default=True,
    )

    path_mode: EnumProperty(
        name="Path Mode",
        description="How texture file paths are written into the .cmod",
        items=[
            ('BASENAME', "Basename",
             "Write only the filename, no directory path (recommended for Celestia)"),
            ('RELATIVE', "Relative",
             "Write a path relative to the exported .cmod file"),
            ('ABSOLUTE', "Absolute",
             "Write the full absolute path (not portable)"),
        ],
        default='BASENAME',
    )

    # --- Performance ---
    worker_count: IntProperty(
        name="Worker Processes",
        description="Number of parallel worker processes for mesh processing. "
                    "0 = use all available CPU cores. "
                    "Only applies when exporting 2+ objects with large meshes.",
        min=0, max=64,
        default=0,
    )

    use_numpy: BoolProperty(
        name="Numpy Acceleration",
        description="Use numpy for fast vertex deduplication. "
                    "Recommended for high-poly meshes (> 20 k faces). "
                    "Requires numpy (included with Blender).",
        default=True,
    )

    # ------------------------------------------------------------------ #
    #  D R A W   P A N E L                                                #
    # ------------------------------------------------------------------ #

    def draw(self, context):
        layout = self.layout
        layout.use_property_split    = True
        layout.use_property_decorate = False

        # ---- Include ----
        header, panel = layout.panel("CMOD_include", default_closed=False)
        header.label(text="Include")
        if panel:
            col = panel.column(heading="Limit to")
            col.prop(self, "use_selection")
            sub = col.row()
            sub.active = not self.use_selection
            sub.prop(self, "use_visible_only")

        # ---- Transform ----
        header, panel = layout.panel("CMOD_transform", default_closed=False)
        header.label(text="Transform")
        if panel:
            panel.prop(self, "global_scale")
            panel.prop(self, "axis_forward")
            panel.prop(self, "axis_up")

        # ---- Geometry ----
        header, panel = layout.panel("CMOD_geometry", default_closed=False)
        header.label(text="Geometry")
        if panel:
            col = panel.column(heading="Mesh")
            col.prop(self, "use_apply_modifiers")
            col.prop(self, "use_triangulate")
            col.prop(self, "use_smooth_groups")

        # ---- Material / Texture ----
        header, panel = layout.panel("CMOD_material", default_closed=False)
        header.label(text="Material")
        if panel:
            col = panel.column(heading="Data")
            col.prop(self, "use_materials")
            col.prop(self, "use_uvs")
            col.prop(self, "path_mode")

        # ---- Performance ----
        header, panel = layout.panel("CMOD_performance", default_closed=False)
        header.label(text="Performance")
        if panel:
            col = panel.column()
            col.prop(self, "worker_count")
            col.prop(self, "use_numpy")
            cpu_n = multiprocessing.cpu_count()
            effective = self.worker_count if self.worker_count > 0 else cpu_n
            col.label(
                text="Detected CPUs: %d  |  Effective workers: %d"
                     % (cpu_n, effective),
                icon='INFO',
            )

    # ------------------------------------------------------------------ #
    #  E X E C U T E                                                       #
    # ------------------------------------------------------------------ #

    def execute(self, context):
        result, message = write_cmod(self.filepath, context, self)
        if result == {'FINISHED'}:
            self.report({'INFO'}, message)
        else:
            self.report({'WARNING'}, message)
        return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def menu_func_export(self, context):
    self.layout.operator(ExportCMOD.bl_idname, text="Celestia Mesh (ascii-cmod)")


def register():
    bpy.utils.register_class(ExportCMOD)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportCMOD)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()
