"""
CMOD Mesh Importer for Blender 5  –  Parallel Edition
Supports Celestia 1.7 ASCII .cmod format



# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# ***** END GPL LICENCE BLOCK *****

bl_info = {
    "name":        "CMOD Mesh Importer (Parallel)",
    "author":      "John Van Vliet",
    "version":     (1, 0, 0),
    "blender":     (5, 0, 1),
    "location":    "File > Import > Celestia Mesh (.cmod)",
    "description": "Import Celestia 1.7 ASCII .cmod mesh files into Blender "
                   "(multi-core geometry preparation)",
    "category":    "Import",
}

# ---------------------------------------------------------------------------
# Guard bpy import so this module stays importable inside worker processes
# spawned by ProcessPoolExecutor (workers do not have bpy available).
# ---------------------------------------------------------------------------
try:
    import bpy
    import mathutils
    from bpy_extras.io_utils import ImportHelper, axis_conversion
    from bpy.props import (
        StringProperty,
        BoolProperty,
        FloatProperty,
        EnumProperty,
        IntProperty,
    )
    from bpy.types import Operator
    _IN_BLENDER = True
except ImportError:                    # running inside a worker process
    _IN_BLENDER = False

import os
import sys
import platform
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np


# ---------------------------------------------------------------------------
# CMOD format constants
# ---------------------------------------------------------------------------

ATTR_COMPONENTS = {
    'f1': 1,
    'f2': 2,
    'f3': 3,
    'f4': 4,
    'ub4': 4,
}

PRIM_TYPES = {'trilist', 'tristrip', 'trifan'}

AXIS_ITEMS = [
    ('X',  "X",  ""),
    ('Y',  "Y",  ""),
    ('Z',  "Z",  ""),
    ('-X', "-X", ""),
    ('-Y', "-Y", ""),
    ('-Z', "-Z", ""),
]


# ===========================================================================
# Tokeniser  (unchanged from original)
# ===========================================================================

class TokenStream:
    """
    Lazy, single-pass tokeniser for ASCII .cmod files.

    Rules
    -----
    * '#' introduces a comment that runs to end-of-line.
    * Quoted strings ("…") are returned as a single token including the
      surrounding quotes so callers can detect and strip them.
    * All other whitespace-delimited runs are plain tokens.
    """

    def __init__(self, filepath):
        self._tokens = []
        self._pos    = 0

        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            raw = fh.read()

        i = 0
        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch == '#':
                while i < n and raw[i] != '\n':
                    i += 1
            elif ch == '"':
                j = i + 1
                while j < n and raw[j] != '"':
                    j += 1
                self._tokens.append(raw[i:j + 1])
                i = j + 1
            elif ch in ' \t\r\n':
                i += 1
            else:
                j = i
                while j < n and raw[j] not in ' \t\r\n"#':
                    j += 1
                self._tokens.append(raw[i:j])
                i = j

    def peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def next(self):
        t = self.peek()
        if t is not None:
            self._pos += 1
        return t

    def expect(self, value):
        t = self.next()
        if t != value:
            raise ValueError(
                "CMOD parse: expected %r but got %r (at token index %d)"
                % (value, t, self._pos))

    def next_float(self):
        return float(self.next())

    def next_int(self):
        return int(self.next())

    def next_string(self):
        t = self.next()
        if t and t.startswith('"') and t.endswith('"'):
            return t[1:-1]
        return t

    def eof(self):
        return self._pos >= len(self._tokens)


# ===========================================================================
# Data classes
# ===========================================================================

class CMODMaterial:
    __slots__ = ('diffuse', 'emissive', 'specular', 'specpower',
                 'opacity', 'texture0', 'emissivemap', 'specularmap', 'normalmap')

    def __init__(self):
        self.diffuse     = (1.0, 1.0, 1.0)
        self.emissive    = None
        self.specular    = (0.5, 0.5, 0.5)
        self.specpower   = 0.0
        self.opacity     = 1.0
        self.texture0    = None
        self.emissivemap = None
        self.specularmap = None
        self.normalmap   = None


class CMODMesh:
    __slots__ = ('vertex_desc', 'raw_verts', 'num_verts', 'primitives')

    def __init__(self):
        self.vertex_desc = []   # [(attr_name, n_components), …]
        self.raw_verts   = []   # flat float list
        self.num_verts   = 0
        self.primitives  = []   # [(prim_type, mat_index, [indices]), …]

    def to_args(self):
        """Return a picklable tuple suitable for _prepare_mesh_geometry()."""
        return (self.vertex_desc, self.raw_verts, self.num_verts, self.primitives)


# ===========================================================================
# Parser  (unchanged from original)
# ===========================================================================

def parse_cmod(filepath):
    ts = TokenStream(filepath)
    materials = []
    meshes    = []

    while not ts.eof():
        token = ts.peek()
        if token == 'material':
            ts.next()
            materials.append(_parse_material(ts))
        elif token == 'mesh':
            ts.next()
            meshes.append(_parse_mesh(ts))
        else:
            ts.next()

    return materials, meshes


def _parse_material(ts):
    mat = CMODMaterial()

    while True:
        tok = ts.peek()
        if tok is None or tok == 'end_material':
            ts.next()
            break
        elif tok == 'diffuse':
            ts.next()
            mat.diffuse = (ts.next_float(), ts.next_float(), ts.next_float())
        elif tok == 'emissive':
            ts.next()
            mat.emissive = (ts.next_float(), ts.next_float(), ts.next_float())
        elif tok == 'specular':
            ts.next()
            mat.specular = (ts.next_float(), ts.next_float(), ts.next_float())
        elif tok == 'specpower':
            ts.next()
            mat.specpower = ts.next_float()
        elif tok == 'opacity':
            ts.next()
            mat.opacity = ts.next_float()
        elif tok == 'texture0':
            ts.next()
            mat.texture0 = ts.next_string()
        elif tok == 'emissivemap':
            ts.next()
            mat.emissivemap = ts.next_string()
        elif tok == 'specularmap':
            ts.next()
            mat.specularmap = ts.next_string()
        elif tok in ('normalmap', 'bumpmap'):
            ts.next()
            mat.normalmap = mat.normalmap or ts.next_string()
        elif tok in ('texture1', 'texture2', 'texture3'):
            ts.next()
            ts.next_string()
        elif tok == 'blend':
            ts.next()
            ts.next()
        else:
            ts.next()

    return mat


def _parse_mesh(ts):
    mesh = CMODMesh()

    while True:
        tok = ts.peek()
        if tok is None or tok == 'end_mesh':
            ts.next()
            break
        elif tok == 'vertexdesc':
            ts.next()
            mesh.vertex_desc = _parse_vertexdesc(ts)
        elif tok == 'vertices':
            ts.next()
            mesh.num_verts = ts.next_int()
            stride         = sum(n for _, n in mesh.vertex_desc)
            total          = mesh.num_verts * stride
            mesh.raw_verts = [ts.next_float() for _ in range(total)]
        elif tok in PRIM_TYPES:
            prim_type = ts.next()
            mat_idx   = ts.next_int()
            num_idx   = ts.next_int()
            indices   = [ts.next_int() for _ in range(num_idx)]
            mesh.primitives.append((prim_type, mat_idx, indices))
        else:
            ts.next()

    return mesh


def _parse_vertexdesc(ts):
    desc = []
    while True:
        tok = ts.peek()
        if tok is None or tok == 'end_vertexdesc':
            ts.next()
            break
        attr_name = ts.next()
        type_str  = ts.next()
        n = ATTR_COMPONENTS.get(type_str)
        if n is not None:
            desc.append((attr_name, n))
    return desc


# ===========================================================================
# Primitive helpers  (unchanged, needed by worker)
# ===========================================================================

def _prim_to_trilist(prim_type, indices):
    if prim_type == 'trilist':
        return list(indices)

    tris = []
    if prim_type == 'tristrip':
        for i in range(len(indices) - 2):
            if i % 2 == 0:
                tris.extend([indices[i],     indices[i + 1], indices[i + 2]])
            else:
                tris.extend([indices[i + 1], indices[i],     indices[i + 2]])
    elif prim_type == 'trifan':
        for i in range(1, len(indices) - 1):
            tris.extend([indices[0], indices[i], indices[i + 1]])

    return tris


# ===========================================================================
# ★  PARALLEL GEOMETRY PREPARATION  (no bpy – safe to run in worker process)
# ===========================================================================

def _prepare_mesh_geometry(args):
    """
    Pure geometry preparation – no bpy dependency.

    Accepts the picklable tuple produced by CMODMesh.to_args() and returns a
    plain dict of lists/numpy arrays ready to feed into bpy foreach_set /
    normals_split_custom_set calls.  Returns None if the mesh has no valid
    triangles.

    Parameters
    ----------
    args : (vertex_desc, raw_verts, num_verts, primitives)

    Returns
    -------
    dict with keys:
        num_verts       int
        pos_flat        list[float]  – len = num_verts * 3
        vi_flat         list[int]    – len = num_faces * 3  (loop vertex indices)
        num_faces       int
        face_mat_slots  list[int]    – len = num_faces
        slot_mat_indices list[int]   – cmod material index per slot
        loop_starts     list[int]    – [0, 3, 6, …]
        uv0_flat        list[float] | None
        uv1_flat        list[float] | None
        col_flat        list[float] | None
        norm_flat       list[float] | None  – len = num_faces * 3 * 3
    """
    vertex_desc, raw_verts, num_verts, primitives = args

    if num_verts == 0 or not raw_verts:
        return None

    # ------------------------------------------------------------------ #
    # 1. Load vertex buffer into a NumPy array  (num_verts × stride)
    # ------------------------------------------------------------------ #
    stride = sum(n for _, n in vertex_desc)
    arr = np.array(raw_verts, dtype=np.float32).reshape(num_verts, stride)

    # Build column-offset map: attr_name → (start_col, width)
    offsets = {}
    off = 0
    for attr_name, n in vertex_desc:
        offsets[attr_name] = (off, n)
        off += n

    def _get(name):
        """Return (num_verts, width) sub-array or None."""
        if name in offsets:
            o, n = offsets[name]
            return arr[:, o:o + n]
        return None

    positions  = _get('position')   # (V, 3)
    normals    = _get('normal')     # (V, 3) or None
    texcoord0  = _get('texcoord0')  # (V, 2) or None
    texcoord1  = _get('texcoord1')  # (V, 2) or None
    color      = _get('color')      # (V, 4) or None

    if positions is None:
        return None

    # ------------------------------------------------------------------ #
    # 2. Triangulate all primitives, build face/slot lists
    # ------------------------------------------------------------------ #
    all_faces      = []   # list of (i0, i1, i2)
    face_mat_slots = []   # parallel per-face local material-slot index
    mat_index_map  = {}   # cmod mat index → local slot index
    slot_mat_indices = [] # local slot index → cmod mat index

    nv = num_verts

    for prim_type, mat_idx, raw_indices in primitives:
        tri_flat = _prim_to_trilist(prim_type, raw_indices)

        if mat_idx not in mat_index_map:
            mat_index_map[mat_idx] = len(slot_mat_indices)
            slot_mat_indices.append(mat_idx)

        local_slot = mat_index_map[mat_idx]

        for t in range(0, len(tri_flat) - 2, 3):
            i0, i1, i2 = tri_flat[t], tri_flat[t + 1], tri_flat[t + 2]
            if i0 == i1 or i1 == i2 or i0 == i2:
                continue
            if i0 >= nv or i1 >= nv or i2 >= nv:
                continue
            all_faces.append((i0, i1, i2))
            face_mat_slots.append(local_slot)

    if not all_faces:
        return None

    faces_np = np.array(all_faces, dtype=np.int32)   # (F, 3)
    num_faces = len(all_faces)
    loops_idx = faces_np.flatten()                    # (F*3,) – vertex per loop

    # ------------------------------------------------------------------ #
    # 3. Flat position list for me.vertices.foreach_set
    # ------------------------------------------------------------------ #
    pos_flat = positions.flatten().tolist()

    # ------------------------------------------------------------------ #
    # 4. UV maps  (flip V: Blender origin is bottom-left, CMOD is top-left)
    # ------------------------------------------------------------------ #
    def _build_uv(uv_attr):
        if uv_attr is None:
            return None
        uv = uv_attr[loops_idx].copy()   # (F*3, 2)
        uv[:, 1] = 1.0 - uv[:, 1]
        return uv.flatten().tolist()

    uv0_flat = _build_uv(texcoord0)
    uv1_flat = _build_uv(texcoord1)

    # ------------------------------------------------------------------ #
    # 5. Vertex colours
    # ------------------------------------------------------------------ #
    col_flat = None
    if color is not None:
        c = color[loops_idx]             # (F*3, ?)
        if c.shape[1] >= 4:
            c = c[:, :4]
        else:
            ones = np.ones((c.shape[0], 4 - c.shape[1]), dtype=np.float32)
            c    = np.hstack([c, ones])
        col_flat = c.flatten().tolist()

    # ------------------------------------------------------------------ #
    # 6. Custom split normals (per-loop, normalised)
    # ------------------------------------------------------------------ #
    norm_flat = None
    if normals is not None:
        n = normals[loops_idx].copy()    # (F*3, 3)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        valid   = (lengths > 1e-6).flatten()
        n[valid]  /= lengths[valid].reshape(-1, 1)
        n[~valid]  = (0.0, 0.0, 1.0)
        norm_flat  = n.flatten().tolist()

    return {
        'num_verts':       num_verts,
        'pos_flat':        pos_flat,
        'vi_flat':         loops_idx.tolist(),
        'num_faces':       num_faces,
        'loop_starts':     list(range(0, num_faces * 3, 3)),
        'face_mat_slots':  face_mat_slots,
        'slot_mat_indices': slot_mat_indices,
        'uv0_flat':        uv0_flat,
        'uv1_flat':        uv1_flat,
        'col_flat':        col_flat,
        'norm_flat':       norm_flat,
    }


def _prepare_all_meshes_parallel(cmod_meshes, n_workers, use_numpy=True):
    """
    Prepare geometry for every mesh, using worker processes where possible.

    On Linux the 'fork' start-method is used, which lets workers inherit the
    parent process state without needing to re-import bpy.  On other platforms
    (Windows, macOS) worker processes would need to import bpy and fail, so we
    fall back to serial execution – NumPy still provides a large speedup.

    When *use_numpy* is False the parallel path is also disabled, since the
    worker function relies entirely on NumPy for array construction.
    """
    args_list = [m.to_args() for m in cmod_meshes]

    # Determine whether multiprocessing is viable on this platform.
    can_fork = (platform.system() == 'Linux')
    use_parallel = (
        can_fork
        and use_numpy
        and len(args_list) > 1
        and n_workers != 1
    )

    if not use_parallel:
        # Serial path (also used as fallback after any executor error).
        return [_prepare_mesh_geometry(args) for args in args_list]

    n = n_workers if n_workers and n_workers > 0 else multiprocessing.cpu_count()
    n = min(n, len(args_list))

    results = [None] * len(args_list)
    try:
        ctx = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            future_to_idx = {
                pool.submit(_prepare_mesh_geometry, args): i
                for i, args in enumerate(args_list)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    print("CMOD Import: worker failed for mesh %d – %s" % (idx, exc))
                    # Retry serially
                    results[idx] = _prepare_mesh_geometry(args_list[idx])
    except Exception as exc:
        print("CMOD Import: ProcessPoolExecutor failed (%s); falling back to serial." % exc)
        results = [_prepare_mesh_geometry(args) for args in args_list]

    return results


# ===========================================================================
# ★  PARALLEL TEXTURE FILE SEARCH  (I/O-bound, ThreadPoolExecutor)
# ===========================================================================

_IMAGE_EXTENSIONS = (
    '',
    '.png', '.jpg', '.jpeg',
    '.tga', '.bmp', '.tiff', '.tif',
    '.dds', '.exr', '.hdr', '.webp',
)


def _find_image_file(filename, search_dirs):
    """
    Search *search_dirs* for *filename*.

    Strategy (in order):
      1. Exact path as given.
      2. Bare basename (strips leading relative path components).
      3. Case-insensitive basename match (needed on Linux).
      4. Steps 1–3 retried with each entry in _IMAGE_EXTENSIONS.

    Returns the first existing absolute path, or None.
    """
    if not filename:
        return None

    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    exts = [ext] + [e for e in _IMAGE_EXTENSIONS if e.lower() != ext.lower()]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            dir_listing = {f.lower(): f for f in os.listdir(d)}
        except OSError:
            dir_listing = {}

        for try_ext in exts:
            rel_name  = os.path.splitext(filename)[0] + try_ext
            candidate = os.path.join(d, rel_name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            bare_name = stem + try_ext
            candidate = os.path.join(d, bare_name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            lower_bare = bare_name.lower()
            if lower_bare in dir_listing:
                candidate = os.path.join(d, dir_listing[lower_bare])
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)

    return None


def _find_all_textures_parallel(cmod_mats, search_dirs):
    """
    Search the filesystem for every texture referenced across all materials
    concurrently.  Filesystem calls release the GIL, so a ThreadPoolExecutor
    gives genuine parallel I/O even on a single CPU core.

    Returns a dict mapping filename → resolved absolute path (or None).
    """
    # Collect unique texture names across all materials.
    texture_names = set()
    for cmat in cmod_mats:
        for fname in (cmat.texture0, cmat.emissivemap,
                      cmat.specularmap, cmat.normalmap):
            if fname:
                texture_names.add(fname)

    if not texture_names:
        return {}

    results = {}
    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(_find_image_file, name, search_dirs): name
            for name in texture_names
        }
        for fut in as_completed(futures):
            name         = futures[fut]
            results[name] = fut.result()

    return results


# ===========================================================================
# Blender texture / material builder  (main thread only)
# ===========================================================================

def _apply_colorspace(img, colorspace):
    FALLBACKS = {
        'sRGB':      ('sRGB - Texture', 'sRGB (Film)',  'Filmic sRGB'),
        'Non-Color': ('Non-Color Data', 'Raw',           'Linear'),
        'Linear':    ('Linear Rec.709', 'Linear BT.709', 'Non-Color'),
    }
    candidates = [colorspace] + list(FALLBACKS.get(colorspace, ()))
    for name in candidates:
        try:
            img.colorspace_settings.name = name
            return
        except TypeError:
            continue


def _load_image(filename, search_dirs, colorspace='sRGB', preresolved_path=None):
    """
    Return (or create) a bpy.types.Image.

    If *preresolved_path* is supplied (from the parallel search phase) the
    filesystem search is skipped entirely.
    """
    if not filename:
        return None

    path = preresolved_path if preresolved_path is not None else _find_image_file(filename, search_dirs)

    if path:
        abs_path = os.path.abspath(path)
        try:
            img = bpy.data.images.load(abs_path, check_existing=True)
            _apply_colorspace(img, colorspace)
            return img
        except Exception as e:
            print("CMOD Import: could not load %r: %s" % (abs_path, e))

    # Placeholder stub
    base = os.path.basename(filename)
    img = bpy.data.images.get(base)
    if img is None:
        img = bpy.data.images.new(base, width=1, height=1)
    img.filepath = filename
    img.source   = 'FILE'
    _apply_colorspace(img, colorspace)
    return img


def _make_tex_node(nodes, image, location, label):
    node          = nodes.new('ShaderNodeTexImage')
    node.image    = image
    node.location = location
    node.label    = label
    return node


def build_blender_material(cmat, mat_index, search_dirs, import_textures,
                            texture_cache=None):
    """
    Build a Principled-BSDF material in Blender from *cmat* (CMODMaterial).

    *texture_cache* is an optional dict mapping filename → resolved path,
    populated by the parallel texture search phase so this function never
    needs to hit the filesystem.
    """
    mat           = bpy.data.materials.new(name="cmod_mat_%d" % mat_index)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out          = nodes.new('ShaderNodeOutputMaterial')
    out.location = (700, 0)

    bsdf          = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    r, g, b = cmat.diffuse
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)

    sr, sg, sb = cmat.specular
    spec_avg   = (sr + sg + sb) / 3.0
    spec_in    = (bsdf.inputs.get('Specular IOR Level')
                  or bsdf.inputs.get('Specular'))
    if spec_in:
        spec_in.default_value = spec_avg

    if cmat.specpower > 0:
        roughness = max(0.0, min(1.0, 1.0 - cmat.specpower / 128.0))
    else:
        roughness = 1.0
    bsdf.inputs['Roughness'].default_value = roughness

    if cmat.opacity < 1.0:
        bsdf.inputs['Alpha'].default_value = cmat.opacity
        mat.blend_method = 'BLEND'

    if cmat.emissive:
        er, eg, eb = cmat.emissive
        strength   = max(er, eg, eb)
        norm_r     = er / max(strength, 1e-6)
        norm_g     = eg / max(strength, 1e-6)
        norm_b     = eb / max(strength, 1e-6)
        emit_col = (bsdf.inputs.get('Emission Color')
                    or bsdf.inputs.get('Emission'))
        emit_str = bsdf.inputs.get('Emission Strength')
        if emit_col:
            emit_col.default_value = (norm_r, norm_g, norm_b, 1.0)
        if emit_str:
            emit_str.default_value = strength

    if import_textures:
        X0        = -700
        UV_OFFSET = -220

        cache = texture_cache or {}

        def wire_uv(tex_node, uv_map_name='UVMap'):
            uv_node          = nodes.new('ShaderNodeUVMap')
            uv_node.uv_map   = uv_map_name
            uv_node.location = (tex_node.location[0] + UV_OFFSET,
                                 tex_node.location[1])
            links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])

        def load(fname, cs):
            return _load_image(fname, search_dirs, cs,
                               preresolved_path=cache.get(fname))

        if cmat.texture0:
            img = load(cmat.texture0, 'sRGB')
            tex = _make_tex_node(nodes, img, (X0, 300), 'Base Color Map')
            wire_uv(tex)
            links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
            if cmat.opacity < 1.0:
                links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])

        if cmat.specularmap:
            img = load(cmat.specularmap, 'Non-Color')
            tex = _make_tex_node(nodes, img, (X0, 0), 'Specular Map')
            wire_uv(tex)
            if spec_in:
                links.new(tex.outputs['Color'], spec_in)

        if cmat.emissivemap:
            img      = load(cmat.emissivemap, 'sRGB')
            tex      = _make_tex_node(nodes, img, (X0, -300), 'Emissive Map')
            wire_uv(tex)
            emit_col = (bsdf.inputs.get('Emission Color')
                        or bsdf.inputs.get('Emission'))
            emit_str = bsdf.inputs.get('Emission Strength')
            if emit_col:
                links.new(tex.outputs['Color'], emit_col)
            if emit_str and emit_str.default_value == 0.0:
                emit_str.default_value = 1.0

        if cmat.normalmap:
            img     = load(cmat.normalmap, 'Non-Color')
            tex     = _make_tex_node(nodes, img, (X0 - 200, -650), 'Normal Map')
            wire_uv(tex)
            nm          = nodes.new('ShaderNodeNormalMap')
            nm.location = (X0 + 120, -650)
            links.new(tex.outputs['Color'], nm.inputs['Color'])
            links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


# ===========================================================================
# ★  Blender mesh builder  (main thread only – consumes pre-prepared data)
# ===========================================================================

def build_blender_mesh(prepped, bl_materials, obj_name, global_matrix):
    """
    Construct a Blender mesh object from a pre-prepared geometry dict.

    All expensive array work was already done by _prepare_mesh_geometry()
    (potentially in a worker process).  This function only makes bpy calls.

    Returns (bpy.types.Object, num_tris_built).
    """
    me = bpy.data.meshes.new(obj_name)

    # ---- Vertices ----
    me.vertices.add(prepped['num_verts'])
    me.vertices.foreach_set('co', prepped['pos_flat'])

    # ---- Loops & polygons ----
    num_faces = prepped['num_faces']
    me.loops.add(num_faces * 3)
    me.polygons.add(num_faces)

    me.loops.foreach_set('vertex_index', prepped['vi_flat'])
    me.polygons.foreach_set('loop_start', prepped['loop_starts'])
    me.polygons.foreach_set('loop_total', [3] * num_faces)

    # ---- Material slots ----
    for cmod_mat_idx in prepped['slot_mat_indices']:
        bl_mat = (bl_materials[cmod_mat_idx]
                  if cmod_mat_idx < len(bl_materials) else None)
        me.materials.append(bl_mat)
    me.polygons.foreach_set('material_index', prepped['face_mat_slots'])

    # ---- UV maps ----
    if prepped['uv0_flat']:
        uv_layer = me.uv_layers.new(name='UVMap')
        uv_layer.data.foreach_set('uv', prepped['uv0_flat'])

    if prepped['uv1_flat']:
        uv_layer1 = me.uv_layers.new(name='UVMap.001')
        uv_layer1.data.foreach_set('uv', prepped['uv1_flat'])

    # ---- Vertex colours ----
    if prepped['col_flat']:
        vcol = me.color_attributes.new(
            name='Col', type='FLOAT_COLOR', domain='CORNER')
        vcol.data.foreach_set('color', prepped['col_flat'])

    # ---- Validate ----
    me.validate(verbose=False)
    me.update()

    # ---- Custom split normals ----
    if prepped['norm_flat']:
        flat = prepped['norm_flat']
        # normals_split_custom_set expects a sequence of (x, y, z) per loop.
        custom_normals = list(zip(flat[0::3], flat[1::3], flat[2::3]))
        me.normals_split_custom_set(custom_normals)
        try:
            me.use_auto_smooth = True
        except AttributeError:
            pass

    for poly in me.polygons:
        poly.use_smooth = True

    me.update()

    # ---- Object ----
    obj              = bpy.data.objects.new(obj_name, me)
    obj.matrix_world = global_matrix

    return obj, num_faces


# ===========================================================================
# ★  Split-by-material helper  (main thread only)
# ===========================================================================

def _split_objects_by_material(context, objects, collection):
    """
    For each object in *objects* that has more than one material slot, use
    Blender's built-in "Separate by Material" operator to break it into one
    object per material.  Objects with a single material (or none) are left
    unchanged.

    All resulting objects are linked into *collection*.

    Returns the new flat list of objects that should replace *objects* in the
    caller's tracking list.
    """
    result = []
    for obj in objects:
        if len(obj.material_slots) <= 1:
            result.append(obj)
            continue

        # Deselect everything, then activate this object alone.
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Enter Edit Mode, select all geometry, separate by material, leave.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.separate(type='MATERIAL')
        except Exception as exc:
            print("CMOD Import: separate-by-material failed for %r – %s" % (obj.name, exc))
        bpy.ops.object.mode_set(mode='OBJECT')

        # Collect the newly created pieces (all selected objects after the op).
        pieces = list(context.selected_objects)

        # Make sure every piece is linked to our target collection and nowhere
        # else (Blender may have linked the new pieces to the scene root).
        for piece in pieces:
            for col in list(piece.users_collection):
                if col is not collection:
                    col.objects.unlink(piece)
            if piece.name not in collection.objects:
                collection.objects.link(piece)

        result.extend(pieces)

    return result


# ===========================================================================
# Main import entry point
# ===========================================================================

def read_cmod(filepath, context, operator):
    """
    Parse *filepath* and populate the active Blender scene.

    Import pipeline
    ---------------
    1. Parse file  (single thread – sequential by nature of the format)
    2. Search for textures  (ThreadPoolExecutor – I/O parallel)
    3. Prepare geometry  (ProcessPoolExecutor on Linux, serial elsewhere)
       Each mesh's vertex unpacking and array building is a separate task.
    4. Build Blender materials  (main thread)
    5. Build Blender mesh objects  (main thread – consumes step-3 output)
    """

    # ---- 1. Parse ----
    try:
        cmod_mats, cmod_meshes = parse_cmod(filepath)
    except Exception as exc:
        return {'CANCELLED'}, "Parse error: %s" % str(exc)

    if not cmod_meshes:
        return {'CANCELLED'}, "No mesh data found in %s" % filepath

    # ---- Shared setup ----
    axis_mat = axis_conversion(
        from_forward=operator.axis_forward,
        from_up=operator.axis_up,
    ).to_4x4()
    scale_mat     = mathutils.Matrix.Scale(operator.global_scale, 4)
    global_matrix = scale_mat @ axis_mat

    cmod_dir    = os.path.dirname(os.path.abspath(filepath))
    search_dirs = [cmod_dir]
    if operator.texture_search_dir:
        d = bpy.path.abspath(operator.texture_search_dir)
        if os.path.isdir(d):
            search_dirs.insert(0, d)

    # ---- 2. Texture search (parallel I/O) ----
    texture_cache = {}
    if operator.import_materials and operator.import_textures:
        texture_cache = _find_all_textures_parallel(cmod_mats, search_dirs)

    # ---- 3. Geometry preparation (parallel CPU) ----
    prepped_list = _prepare_all_meshes_parallel(
        cmod_meshes, operator.worker_count, operator.use_numpy)

    # ---- 4. Build materials (main thread) ----
    bl_materials = []
    if operator.import_materials:
        for i, cmat in enumerate(cmod_mats):
            bl_mat = build_blender_material(
                cmat, i, search_dirs,
                operator.import_textures,
                texture_cache=texture_cache)
            bl_materials.append(bl_mat)
    else:
        bl_materials = [None] * len(cmod_mats)

    # ---- Collection ----
    if operator.use_collection:
        col_name   = os.path.splitext(os.path.basename(filepath))[0]
        collection = bpy.data.collections.new(col_name)
        context.scene.collection.children.link(collection)
    else:
        collection = context.scene.collection

    # ---- 5. Build mesh objects (main thread) ----
    base_name   = os.path.splitext(os.path.basename(filepath))[0]
    new_objects = []
    total_tris  = 0

    for i, (cmesh, prepped) in enumerate(zip(cmod_meshes, prepped_list)):
        if prepped is None:
            print("CMOD Import: skipping mesh %d — no valid geometry." % i)
            continue

        obj_name = (base_name if len(cmod_meshes) == 1
                    else "%s_%03d" % (base_name, i))
        try:
            obj, ntris = build_blender_mesh(
                prepped, bl_materials, obj_name, global_matrix)
        except Exception as exc:
            import traceback
            print("CMOD Import: skipping mesh %d — %s" % (i, exc))
            traceback.print_exc()
            continue

        collection.objects.link(obj)
        new_objects.append(obj)
        total_tris += ntris

    if not new_objects:
        return {'CANCELLED'}, "Failed to build any valid mesh objects."

    # ---- Optionally split by material (before any merge) ----
    if operator.split_by_material and not operator.merge_meshes:
        new_objects = _split_objects_by_material(context, new_objects, collection)

    # ---- Optionally merge ----
    if operator.merge_meshes and len(new_objects) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in new_objects:
            obj.select_set(True)
        context.view_layer.objects.active = new_objects[0]
        with context.temp_override(
                active_object=new_objects[0],
                selected_objects=new_objects):
            bpy.ops.object.join()
        new_objects = [context.view_layer.objects.active]

    # ---- Select results ----
    bpy.ops.object.select_all(action='DESELECT')
    for obj in new_objects:
        obj.select_set(True)
    context.view_layer.objects.active = new_objects[0]

    # ---- Zoom to fit ----
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            with context.temp_override(area=area):
                try:
                    bpy.ops.view3d.view_selected()
                except Exception:
                    pass
            break

    msg = ("Imported %s — %d material(s), %d mesh(es), %d triangle(s)"
           % (os.path.basename(filepath),
              len(cmod_mats), len(new_objects), total_tris))
    return {'FINISHED'}, msg


# ===========================================================================
# Helper operator – texture-directory browser
# ===========================================================================
# Blender only allows one file-selector dialog at a time.  Because ImportHelper
# already opens a browser for the .cmod file, using subtype='DIR_PATH' on the
# texture_search_dir property would conflict and raise:
#   "Cannot activate a file selector dialog, one already open"
#
# The fix: subtype='NONE' on the property (removes the broken folder icon),
# plus this small independent operator that opens its own browser window and
# writes the chosen path back into the WindowManager so the import operator
# ===========================================================================
# Operator / UI
# ===========================================================================
# ===========================================================================

class ImportCMOD(Operator, ImportHelper):
    """Import a Celestia ASCII .cmod mesh file"""
    bl_idname  = "import_mesh.cmod"
    bl_label   = "Import CMOD"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".cmod"

    filter_glob: StringProperty(
        default="*.cmod",
        options={'HIDDEN'},
        maxlen=255,
    )

    # --- Transform ---
    global_scale: FloatProperty(
        name="Scale",
        description="Uniform scale applied to the imported geometry",
        min=0.0001, max=10000.0,
        soft_min=0.01, soft_max=100.0,
        default=1.0,
        step=10,
        precision=4,
    )

    axis_forward: EnumProperty(
        name="Forward",
        description="The axis in the .cmod file that points forward",
        items=AXIS_ITEMS,
        default='X',
    )

    axis_up: EnumProperty(
        name="Up",
        description="The axis in the .cmod file that points up",
        items=AXIS_ITEMS,
        default='Z',
    )

    # --- Geometry ---
    merge_meshes: BoolProperty(
        name="Merge Mesh Blocks",
        description="Join all mesh blocks into a single Blender object "
                    "(overrides Split by Material)",
        default=False,
    )

    split_by_material: BoolProperty(
        name="Split by Material",
        description=(
            "Create a separate Blender object for every distinct material group "
            "within each mesh block.  Useful when a single .cmod mesh block "
            "references multiple materials and you want to work on them "
            "independently.  Ignored when 'Merge Mesh Blocks' is enabled."
        ),
        default=False,
    )

    use_collection: BoolProperty(
        name="Import into New Collection",
        description="Place imported objects in a new collection named after the .cmod file",
        default=True,
    )

    # --- Material ---
    import_materials: BoolProperty(
        name="Import Materials",
        description="Build Principled BSDF materials from the .cmod material blocks",
        default=True,
    )

    import_textures: BoolProperty(
        name="Load Textures",
        description="Attempt to load texture images referenced in material blocks",
        default=True,
    )

    texture_search_dir: StringProperty(
        name="Texture Search Dir",
        description="Extra directory searched for texture images (type, paste, "
                    "or click the folder icon to browse)",
        default="",
        subtype='DIR_PATH',
    )

    # --- Performance ---
    worker_count: IntProperty(
        name="Worker Processes",
        description=(
            "Number of worker processes for parallel geometry preparation "
            "(Linux only; 0 = use all CPU cores; 1 = serial). "
            "On Windows/macOS this setting is ignored and serial processing "
            "is used – NumPy still accelerates array building."
        ),
        min=0, max=64,
        default=0,
    )

    use_numpy: BoolProperty(
        name="Numpy Acceleration",
        description=(
            "Use NumPy for accelerated geometry array building. "
            "Recommended for large meshes. "
            "Parallel processing requires NumPy; disabling this forces "
            "serial execution regardless of the Worker Processes setting. "
            "Requires NumPy (included with Blender)."
        ),
        default=True,
    )

    # --- Draw panel ---
    def draw(self, context):
        layout = self.layout
        layout.use_property_split    = True
        layout.use_property_decorate = False

        header, panel = layout.panel("CMOD_IMP_transform", default_closed=False)
        header.label(text="Transform")
        if panel:
            panel.prop(self, "global_scale")
            panel.prop(self, "axis_forward")
            panel.prop(self, "axis_up")

        header, panel = layout.panel("CMOD_IMP_geometry", default_closed=False)
        header.label(text="Geometry")
        if panel:
            panel.prop(self, "merge_meshes")
            row = panel.row()
            row.prop(self, "split_by_material")
            row.active = not self.merge_meshes   # grey out when Merge is on
            panel.prop(self, "use_collection")

        header, panel = layout.panel("CMOD_IMP_material", default_closed=False)
        header.label(text="Material")
        if panel:
            col = panel.column(heading="Data")
            col.prop(self, "import_materials")
            sub = col.column()
            sub.active = self.import_materials
            sub.prop(self, "import_textures")
            sub2 = sub.column()
            sub2.active = self.import_textures
            sub2.prop(self, "texture_search_dir", text="Texture Dir")

        header, panel = layout.panel("CMOD_IMP_perf", default_closed=False)
        header.label(text="Performance")
        if panel:
            col = panel.column()
            col.prop(self, "worker_count")
            col.prop(self, "use_numpy")
            cpu_n     = multiprocessing.cpu_count()
            effective = self.worker_count if self.worker_count > 0 else cpu_n
            col.label(
                text="Detected CPUs: %d  |  Effective workers: %d"
                     % (cpu_n, effective),
                icon='INFO',
            )
            if platform.system() != 'Linux':
                row = col.row()
                row.alert = True
                row.label(text="Multi-process only available on Linux",
                          icon='INFO')

    def execute(self, context):
        result, message = read_cmod(self.filepath, context, self)
        level = 'INFO' if result == {'FINISHED'} else 'WARNING'
        self.report({level}, message)
        return result


# ===========================================================================
# Registration
# ===========================================================================

def menu_func_import(self, context):
    self.layout.operator(ImportCMOD.bl_idname, text="Celestia Mesh (ascii-cmod)")


def register():
    bpy.utils.register_class(ImportCMOD)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(ImportCMOD)


if __name__ == "__main__":
    register()
