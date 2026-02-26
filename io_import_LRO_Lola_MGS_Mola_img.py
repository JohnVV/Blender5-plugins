# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "LRO Lola & MGS Mola img/DEM Importer",
    "author": "Valter Battioli (ValterVB)-for Blender 2.58 - updated to blender 5 by : John Van Vliet",
    "version": (2, 0, 0),
    "blender": (5, 0, 0),
    "location": "3D Viewport > Sidebar > LRO/MGS",
    "description": "Import DTM from LRO Lola and MGS Mola",
    "warning": "May consume a lot of memory",
    "doc_url": "https://docs.blender.org/",
    "category": "Import-Export",
}



import bpy
import os
import os.path
import math
import array
import numpy as np
import mathutils
from mathutils import Vector
from bpy_extras.io_utils import axis_conversion

TO_RAD = math.pi / 180  # From degrees to radians

# Turning off relative path - it causes errors if enabled
# NOTE: user_preferences was renamed to preferences in Blender 2.80
if bpy.context.preferences.filepaths.use_relative_paths:
    bpy.context.preferences.filepaths.use_relative_paths = False


# A very simple "bridge" tool.
# Connects two equally long vertex rows with faces.
# Returns a list of the new faces (list of lists)
#
# vertIdx1 ... First vertex list (list of vertex indices).
# vertIdx2 ... Second vertex list (list of vertex indices).
# closed ... Creates a loop (first & last are closed).
# flipped ... Invert the normal of the face(s).
#
# Note: You can set vertIdx1 to a single vertex index to create a fan/star
#       of faces.
# Note: If both vertex idx list are the same length they have to have at
#       least 2 vertices.
def createFaces(vertIdx1, vertIdx2, closed=False, flipped=False):
    faces = []

    if not vertIdx1 or not vertIdx2:
        return None

    if len(vertIdx1) < 2 and len(vertIdx2) < 2:
        return None

    fan = False
    if len(vertIdx1) != len(vertIdx2):
        if len(vertIdx1) == 1 and len(vertIdx2) > 1:
            fan = True
        else:
            return None

    total = len(vertIdx2)

    if closed:
        # Bridge the start with the end.
        if flipped:
            face = [vertIdx1[0], vertIdx2[0], vertIdx2[total - 1]]
            if not fan:
                face.append(vertIdx1[total - 1])
            faces.append(face)
        else:
            face = [vertIdx2[0], vertIdx1[0]]
            if not fan:
                face.append(vertIdx1[total - 1])
            face.append(vertIdx2[total - 1])
            faces.append(face)

    # Bridge the rest of the faces.
    for num in range(total - 1):
        if flipped:
            if fan:
                face = [vertIdx2[num], vertIdx1[0], vertIdx2[num + 1]]
            else:
                face = [vertIdx2[num], vertIdx1[num],
                        vertIdx1[num + 1], vertIdx2[num + 1]]
            faces.append(face)
        else:
            if fan:
                face = [vertIdx1[0], vertIdx2[num], vertIdx2[num + 1]]
            else:
                face = [vertIdx1[num], vertIdx2[num],
                        vertIdx2[num + 1], vertIdx1[num + 1]]
            faces.append(face)
    return faces


# Utility Functions ********************************************************

# Input: Latitude  Output: Number of the line (1 to n)
def LatToLine(Latitude):
    tmpLines = round((MAXIMUM_LATITUDE - Latitude) * MAP_RESOLUTION + 1.0)
    if tmpLines > LINES:
        tmpLines = LINES
    return tmpLines


# Input: Number of the line (1 to n)  Output: Latitude
def LineToLat(Line):
    if MAP_RESOLUTION == 0:
        return 0
    else:
        return float(MAXIMUM_LATITUDE - (Line - 1) / MAP_RESOLUTION)


# Input: Longitude  Output: Number of the point (1 to n)
def LongToPoint(Longitude):
    tmpPoints = round((Longitude - WESTERNMOST_LONGITUDE) *
                      MAP_RESOLUTION + 1.0)
    if tmpPoints > LINE_SAMPLES:
        tmpPoints = LINE_SAMPLES
    return tmpPoints


# Input: Number of the Point (1 to n)  Output: Longitude
def PointToLong(Point):
    if MAP_RESOLUTION == 0:
        return 0
    else:
        return float(WESTERNMOST_LONGITUDE + (Point - 1) / MAP_RESOLUTION)


# Input: Latitude  Output: Nearest real Latitude on grid
def RealLat(Latitude):
    return float(LineToLat(LatToLine(Latitude)))


# Input: Longitude  Output: Nearest real Longitude on grid
def RealLong(Longitude):
    return float(PointToLong(LongToPoint(Longitude)))

# **************************************************************************


def MakeMaterialMars(obj):
    """Create a Mars-like material using Blender's node-based pipeline."""
    mat = bpy.data.materials.new("Mars")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.426, 0.213, 0.136, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.01
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.append(mat)


def MakeMaterialMoon(obj):
    """Create a Moon-like material using Blender's node-based pipeline."""
    mat = bpy.data.materials.new("Moon")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.426, 0.426, 0.426, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.01
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.append(mat)


# Read the LBL file
def ReadLabel(FileName):
    global FileAndPath
    global LINES, LINE_SAMPLES, SAMPLE_TYPE, SAMPLE_BITS, UNIT, MAP_RESOLUTION
    global MAXIMUM_LATITUDE, MINIMUM_LATITUDE, WESTERNMOST_LONGITUDE
    global EASTERNMOST_LONGITUDE, SCALING_FACTOR, OFFSET, RadiusUM, TARGET_NAME
    global Message

    if FileName == '':
        return
    LINES = LINE_SAMPLES = SAMPLE_BITS = MAP_RESOLUTION = 0
    MAXIMUM_LATITUDE = MINIMUM_LATITUDE = 0.0
    WESTERNMOST_LONGITUDE = EASTERNMOST_LONGITUDE = 0.0
    OFFSET = SCALING_FACTOR = 0.0
    SAMPLE_TYPE = UNIT = TARGET_NAME = RadiusUM = Message = ""

    FileAndPath = FileName
    FileAndExt = os.path.splitext(FileAndPath)
    try:
        # Check for UNIX case sensitivity:
        # If the extension chosen by the user is uppercase, open uppercase LBL,
        # and vice versa.
        if FileAndExt[1].isupper():
            f = open(FileAndExt[0] + ".LBL", 'r')
        else:
            f = open(FileAndExt[0] + ".lbl", 'r')
        Message = ""
    except Exception:
        Message = "FILE LBL NOT AVAILABLE OR YOU HAVEN'T SELECTED A FILE"
        return

    block = ""
    OFFSET = 0
    A_AXIS_RADIUS = B_AXIS_RADIUS = C_AXIS_RADIUS = 0.0

    for line in f:
        tmp = line.split("=")
        if tmp[0].strip() == "OBJECT" and tmp[1].strip() == "IMAGE":
            block = "IMAGE"
        elif tmp[0].strip() == "OBJECT" and tmp[1].strip() == "IMAGE_MAP_PROJECTION":
            block = "IMAGE_MAP_PROJECTION"
        elif tmp[0].strip() == "END_OBJECT" and tmp[1].strip() == "IMAGE":
            block = ""
        elif tmp[0].strip() == "END_OBJECT" and tmp[1].strip() == "IMAGE_MAP_PROJECTION":
            block = ""
        elif tmp[0].strip() == "TARGET_NAME":
            block = ""
            TARGET_NAME = tmp[1].strip()

        if block == "IMAGE":
            if line.find("LINES") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                LINES = int(tmp[1].strip())
            elif line.find("LINE_SAMPLES") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                LINE_SAMPLES = int(tmp[1].strip())
            elif line.find("UNIT") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                UNIT = tmp[1].strip()
            elif line.find("SAMPLE_TYPE") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                SAMPLE_TYPE = tmp[1].strip()
            elif line.find("SAMPLE_BITS") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                SAMPLE_BITS = int(tmp[1].strip())
            elif line.find("SCALING_FACTOR") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                SCALING_FACTOR = float(tmp[0].replace(" ", ""))
            elif line.find("OFFSET") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                if tmp[0].find("OFFSET") != -1 and len(tmp) > 1:
                    tmp = tmp[1].split("<")
                    tmp[0] = tmp[0].replace(".", "")
                    OFFSET = float(tmp[0].replace(" ", ""))

        elif block == "IMAGE_MAP_PROJECTION":
            if line.find("A_AXIS_RADIUS") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                A_AXIS_RADIUS = float(tmp[0].replace(" ", ""))
                RadiusUM = tmp[1].rstrip().replace(">", "")
            elif line.find("B_AXIS_RADIUS") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                B_AXIS_RADIUS = float(tmp[0].replace(" ", ""))
            elif line.find("C_AXIS_RADIUS") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                C_AXIS_RADIUS = float(tmp[0].replace(" ", ""))
            elif line.find("MAXIMUM_LATITUDE") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                MAXIMUM_LATITUDE = float(tmp[0].replace(" ", ""))
            elif line.find("MINIMUM_LATITUDE") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                MINIMUM_LATITUDE = float(tmp[0].replace(" ", ""))
            elif line.find("WESTERNMOST_LONGITUDE") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                WESTERNMOST_LONGITUDE = float(tmp[0].replace(" ", ""))
            elif line.find("EASTERNMOST_LONGITUDE") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                EASTERNMOST_LONGITUDE = float(tmp[0].replace(" ", ""))
            elif line.find("MAP_RESOLUTION") != -1 and not line.startswith("/*"):
                tmp = line.split("=")
                tmp = tmp[1].split("<")
                MAP_RESOLUTION = float(tmp[0].replace(" ", ""))

    f.close()  # was "f.close" (missing call) — fixed

    MAXIMUM_LATITUDE = MAXIMUM_LATITUDE - 1 / MAP_RESOLUTION / 2
    MINIMUM_LATITUDE = MINIMUM_LATITUDE + 1 / MAP_RESOLUTION / 2
    WESTERNMOST_LONGITUDE = WESTERNMOST_LONGITUDE + 1 / MAP_RESOLUTION / 2
    EASTERNMOST_LONGITUDE = EASTERNMOST_LONGITUDE - 1 / MAP_RESOLUTION / 2

    if OFFSET == 0:  # When OFFSET isn't available use the mean of the radii
        OFFSET = (A_AXIS_RADIUS + B_AXIS_RADIUS + C_AXIS_RADIUS) / 3
    else:
        OFFSET = OFFSET / 1000  # Convert m to Km

    if SCALING_FACTOR == 0:
        SCALING_FACTOR = 1.0  # When unavailable default to 1


def update_fpath(self, context):
    global start_up
    start_up = False
    ReadLabel(bpy.context.scene.fpath)
    if Message != "":
        start_up = True
    else:
        typ = bpy.types.Scene
        var = bpy.props
        typ.FromLat = var.FloatProperty(
            description="From Latitude",
            min=float(MINIMUM_LATITUDE),
            max=float(MAXIMUM_LATITUDE),
            precision=3,
            default=0.0,
        )
        typ.ToLat = var.FloatProperty(
            description="To Latitude",
            min=float(MINIMUM_LATITUDE),
            max=float(MAXIMUM_LATITUDE),
            precision=3,
        )
        typ.FromLong = var.FloatProperty(
            description="From Longitude",
            min=float(WESTERNMOST_LONGITUDE),
            max=float(EASTERNMOST_LONGITUDE),
            precision=3,
        )
        typ.ToLong = var.FloatProperty(
            description="To Longitude",
            min=float(WESTERNMOST_LONGITUDE),
            max=float(EASTERNMOST_LONGITUDE),
            precision=3,
        )
        typ.Scale = var.IntProperty(description="Scale", min=1, max=100, default=1)
        typ.Magnify = var.BoolProperty(description="Magnify", default=False)



# Import the data and draw the planet
class Import(bpy.types.Operator):
    bl_idname = 'import.lro_and_mgs'
    bl_label = 'Start Import'
    bl_description = 'Import the data'

    def execute(self, context):
        From_Lat = RealLat(bpy.context.scene.FromLat)
        To_Lat = RealLat(bpy.context.scene.ToLat)
        From_Long = RealLong(bpy.context.scene.FromLong)
        To_Long = RealLong(bpy.context.scene.ToLong)
        BlenderScale = bpy.context.scene.Scale
        Exag = bpy.context.scene.Magnify
        Vertex = []
        Faces = []
        FirstRow = []
        SecondRow = []

        print('*** Start create vertex ***')
        FileAndPath = bpy.context.scene.fpath
        FileAndExt = os.path.splitext(FileAndPath)
        # Case-sensitivity fix for UNIX:
        if FileAndExt[1].isupper():
            FileName = FileAndExt[0] + ".IMG"
        else:
            FileName = FileAndExt[0] + ".img"

        # ------------------------------------------------------------------
        # Phase 1 – Read all altitude rows into a 2-D NumPy array
        #
        # File I/O is inherently sequential (each row is at a calculated
        # byte offset), so we read row-by-row but use np.frombuffer() to
        # parse the raw bytes directly into int16 arrays without any
        # intermediate Python array.array or byteswap step.
        # ------------------------------------------------------------------
        # numpy dtype encodes both element size and byte order, so no
        # manual byteswap() is needed.
        np_dtype = np.dtype(">i2") if SAMPLE_TYPE == "MSB_INTEGER" else np.dtype("<i2")

        bytes_per_sample = SAMPLE_BITS // 8
        SkipFirstPoint  = int((LongToPoint(From_Long) - 1) * bytes_per_sample)
        PointsToRead    = int((LongToPoint(To_Long) - LongToPoint(From_Long) + 1) * bytes_per_sample)
        SkipLastPoint   = int(LINE_SAMPLES * bytes_per_sample - PointsToRead - SkipFirstPoint)

        f = open(FileName, 'rb')
        f.seek(int((LatToLine(From_Lat) - 1) * LINE_SAMPLES * bytes_per_sample), 1)

        altitude_rows = []
        LatToRead = From_Lat
        while LatToRead >= To_Lat - 1e-9:          # epsilon guards float drift
            f.seek(SkipFirstPoint, 1)
            raw = f.read(PointsToRead)
            altitude_rows.append(np.frombuffer(raw, dtype=np_dtype).astype(np.float64))
            f.seek(SkipLastPoint, 1)
            LatToRead -= 1.0 / MAP_RESOLUTION

        f.close()

        # Stack rows → 2-D array  shape: (n_rows, n_cols)
        altitudes_2d = np.stack(altitude_rows)
        n_rows, n_cols = altitudes_2d.shape
        print(f'    Read {n_rows} x {n_cols} altitude grid from file')

        # ------------------------------------------------------------------
        # Phase 2 – Vectorized X/Y/Z computation (NumPy, multi-core BLAS)
        #
        # All trig and radius maths operate on full 2-D arrays at once.
        # NumPy's C-compiled ufuncs release the GIL and numpy's BLAS/LAPACK
        # backend automatically distributes the work across available CPU
        # cores — no manual process management required.
        # ------------------------------------------------------------------
        step = 1.0 / MAP_RESOLUTION
        # lat descends from From_Lat; lon ascends from From_Long
        lat_1d = From_Lat  - np.arange(n_rows) * step   # shape (n_rows,)
        lon_1d = From_Long + np.arange(n_cols) * step   # shape (n_cols,)

        # meshgrid with 'ij' indexing → both grids are (n_rows, n_cols),
        # matching the row-major order of altitudes_2d.
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')

        lat_rad = lat_2d * (np.pi / 180.0)
        lon_rad = lon_2d * (np.pi / 180.0)

        height_exag = bpy.context.scene.HeightExaggeration
        if Exag:
            tmp_radius = (altitudes_2d * height_exag / SCALING_FACTOR / 1000.0 + OFFSET) / BlenderScale
        else:
            tmp_radius = (altitudes_2d * height_exag * SCALING_FACTOR / 1000.0 + OFFSET) / BlenderScale

        cos_lat       = np.cos(lat_rad)
        current_radius = tmp_radius * cos_lat

        X = current_radius * np.sin(lon_rad)   # shape (n_rows, n_cols)
        Y = tmp_radius     * np.sin(lat_rad)
        Z = current_radius * np.cos(lon_rad)

        # Interleave into (n_rows * n_cols, 3) in row-major order,
        # then convert each row to a mathutils Vector.
        coords = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
        Vertex = [Vector(v.tolist()) for v in coords]

        del altitudes_2d, altitude_rows, coords, X, Y, Z, tmp_radius
        print('*** End create Vertex   ***')

        print('*** Start create faces ***')
        LinesToRead = int(LatToLine(To_Lat) - LatToLine(From_Lat) + 1)
        PointsToRead = int(LongToPoint(To_Long) - LongToPoint(From_Long) + 1)

        for Point in range(0, PointsToRead):
            FirstRow.append(Point)
            SecondRow.append(Point + PointsToRead)
        if int(PointsToRead) == LINE_SAMPLES:
            FaceTemp = createFaces(FirstRow, SecondRow, closed=True, flipped=True)
        else:
            FaceTemp = createFaces(FirstRow, SecondRow, closed=False, flipped=True)
        Faces.extend(FaceTemp)

        FaceTemp = []
        for Line in range(1, (LinesToRead - 1)):
            FirstRow = SecondRow
            SecondRow = []
            FacesTemp = []
            for Point in range(0, PointsToRead):
                SecondRow.append(Point + (Line + 1) * PointsToRead)
            if int(PointsToRead) == LINE_SAMPLES:
                FaceTemp = createFaces(FirstRow, SecondRow, closed=True, flipped=True)
            else:
                FaceTemp = createFaces(FirstRow, SecondRow, closed=False, flipped=True)
            Faces.extend(FaceTemp)
        del FaceTemp
        print('*** End create faces   ***')

        print('*** Start draw ***')
        mesh = bpy.data.meshes.new(TARGET_NAME)
        mesh.from_pydata(Vertex, [], Faces)
        del Faces
        del Vertex
        mesh.update()
        ob_new = bpy.data.objects.new(TARGET_NAME, mesh)
        ob_new.data = mesh

        # 2.80+: link to the active collection instead of scene.objects.link
        bpy.context.collection.objects.link(ob_new)
        # 2.80+: use view_layer.objects.active instead of scene.objects.active
        bpy.context.view_layer.objects.active = ob_new
        # 2.80+: use select_set() instead of ob.select = True
        ob_new.select_set(True)

        # --- Apply Transform (Scale + Axis Orientation) ---
        # The mesh is built in a Y-up / Z-forward coordinate space:
        #   X = radius * sin(lon)   (east)
        #   Y = radius * sin(lat)   (up / altitude)
        #   Z = radius * cos(lon)   (prime-meridian forward)
        # axis_conversion() returns a rotation matrix that maps that native
        # space into whatever Up / Forward the user requested.
        up_axis     = bpy.context.scene.UpAxis
        fwd_axis    = bpy.context.scene.ForwardAxis
        extra_scale = bpy.context.scene.TransformScale

        # Map our enum values to the string tokens axis_conversion expects.
        # NEGATIVE_X/Y/Z → '-X'/'-Y'/'-Z'
        def _ax(token):
            return token.replace('NEGATIVE_', '-')

        conv_mat = axis_conversion(
            from_forward='Z',
            from_up='Y',
            to_forward=_ax(fwd_axis),
            to_up=_ax(up_axis),
        ).to_4x4()

        scale_mat = mathutils.Matrix.Scale(extra_scale, 4)
        ob_new.matrix_world = conv_mat @ scale_mat
        print('*** End draw   ***')

        print('*** Start Smooth ***')
        bpy.ops.object.shade_smooth()
        print('*** End Smooth   ***')

        if TARGET_NAME == "MOON":
            MakeMaterialMoon(ob_new)
        elif TARGET_NAME == "MARS":
            MakeMaterialMars(ob_new)

        print('*** FINISHED ***')
        return {'FINISHED'}


# User interface
class Img_Importer(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    # 2.80+: "TOOL_PROPS" is gone; use "UI" for the N-panel sidebar
    bl_region_type = "UI"
    bl_category = "LRO/MGS"
    bl_label = "LRO Lola & MGS Mola IMG Importer"

    def draw(self, context):
        layout = self.layout
        if start_up:
            layout.prop(context.scene, "fpath")
            col = layout.column()
            if Message != "":
                # 2.80+: label() requires keyword argument text=
                col.label(text="Message: " + Message)
        else:
            col = layout.column()

            split = col.split(factor=0.5)
            split.label(text="Minimum Latitude: " + str(MINIMUM_LATITUDE) + " deg")
            split.label(text="Maximum Latitude: " + str(MAXIMUM_LATITUDE) + " deg")

            split = col.split(factor=0.5)
            split.label(text="Westernmost Longitude: " + str(WESTERNMOST_LONGITUDE) + " deg")
            split.label(text="Easternmost Longitude: " + str(EASTERNMOST_LONGITUDE) + " deg")

            split = col.split(factor=0.5)
            split.label(text="Lines: " + str(LINES))
            split.label(text="Line samples: " + str(LINE_SAMPLES))

            split = col.split(factor=0.5)
            split.label(text="Sample type: " + str(SAMPLE_TYPE))
            split.label(text="Sample bits: " + str(SAMPLE_BITS))

            split = col.split(factor=0.5)
            split.label(text="Unit: " + UNIT)
            split.label(text="Map resolution: " + str(MAP_RESOLUTION) + " pix/deg")

            split = col.split(factor=0.5)
            split.label(text="Radius: " + str(OFFSET) + " " + RadiusUM)
            split.label(text="Scale: " + str(SCALING_FACTOR))

            split = col.split(factor=0.5)
            split.label(text="Target: ")
            split.label(text=TARGET_NAME)

            col = layout.column()
            split = col.split(factor=0.5)
            # 2.80+: prop() text label must use keyword text=
            split.prop(context.scene, "FromLat", text="Northernmost Lat.")
            split.prop(context.scene, "ToLat", text="Southernmost Lat.")
            if bpy.context.scene.FromLat < bpy.context.scene.ToLat:
                col = layout.column()
                col.label(text="Warning: Northernmost must be greater than Southernmost")

            col = layout.column()
            split = col.split(factor=0.5)
            split.prop(context.scene, "FromLong", text="Westernmost Long.")
            split.prop(context.scene, "ToLong", text="Easternmost Long.")
            if bpy.context.scene.FromLong > bpy.context.scene.ToLong:
                col = layout.column()
                col.label(text="Warning: Easternmost must be greater than Westernmost")

            col = layout.column()
            split = col.split(factor=0.5)
          #  split.prop(context.scene, "Scale", text="Scale")
          # split.prop(context.scene, "Magnify", text="Magnify (x4)")

            if bpy.context.scene.fpath != "":
                col = layout.column()
               # col.label(text="1 Blender unit = " + str(bpy.context.scene.Scale) + RadiusUM)

            if Message != "":
                col = layout.column()
                col.label(text="Message: " + Message)

            if bpy.context.scene.fpath.upper().endswith(("IMG", "LBL")):
                VertNumbers = (
                    ((RealLat(bpy.context.scene.FromLat) - RealLat(bpy.context.scene.ToLat)) * MAP_RESOLUTION) + 1
                ) * (
                    (RealLong(bpy.context.scene.ToLong) - RealLong(bpy.context.scene.FromLong)) * MAP_RESOLUTION + 1
                )
            else:
                VertNumbers = 0

            # Need ≥4 vertices, ≥2 rows, ≥2 points to import
            if (
                VertNumbers > 3
                and (RealLat(bpy.context.scene.FromLat) > RealLat(bpy.context.scene.ToLat))
                and (RealLong(bpy.context.scene.FromLong) < RealLong(bpy.context.scene.ToLong))
            ):
                col = layout.column()
                split = col.split(factor=0.5)
                split.label(text="Map resolution on the equator: ")
                split.label(
                    text=str(2 * math.pi * OFFSET / 360 / MAP_RESOLUTION) + " " + RadiusUM + "/pix"
                )
                col = layout.column()
                split = col.split(factor=0.5)
                split.label(text="Real Northernmost Lat.: " + str(RealLat(bpy.context.scene.FromLat)) + " deg")
                split.label(text="Real Southernmost Lat.: " + str(RealLat(bpy.context.scene.ToLat)) + " deg")
                split = col.split(factor=0.5)
                split.label(text="Real Westernmost Long.: " + str(RealLong(bpy.context.scene.FromLong)) + " deg")
                split.label(text="Real Easternmost Long.: " + str(RealLong(bpy.context.scene.ToLong)) + " deg")
                split = col.split(factor=0.5)
                split.label(text="Number of vertices to import: " + str(int(VertNumbers)))
                col.separator()

                # --- Transform section ---
                box = layout.box()
                box.label(text="Transform", icon='OBJECT_ORIGIN')
                row = box.row()
                row.prop(context.scene, "TransformScale", text="Scale")
                row = box.row()
                row.prop(context.scene, "HeightExaggeration", text="Height Exaggeration")
                split = box.split(factor=0.5)
                split.prop(context.scene, "UpAxis", text="Up Axis")
                split.prop(context.scene, "ForwardAxis", text="Forward Axis")

                col = layout.column()
                col.separator()
                col.operator('import.lro_and_mgs', text='Import')
                col.separator()
                col.operator('import.reset', text='Reset')


# Reset the UI
class Reset(bpy.types.Operator):
    bl_idname = 'import.reset'
    bl_label = 'Reset'

    def execute(self, context):
        clear_properties()
        return {'FINISHED'}


def initialize():
    global MAXIMUM_LATITUDE, MINIMUM_LATITUDE
    global WESTERNMOST_LONGITUDE, EASTERNMOST_LONGITUDE
    global LINES, LINE_SAMPLES, SAMPLE_BITS, MAP_RESOLUTION
    global OFFSET, SCALING_FACTOR
    global SAMPLE_TYPE, UNIT, TARGET_NAME, RadiusUM, Message
    global start_up

    LINES = LINE_SAMPLES = SAMPLE_BITS = MAP_RESOLUTION = 0
    MAXIMUM_LATITUDE = MINIMUM_LATITUDE = 0.0
    WESTERNMOST_LONGITUDE = EASTERNMOST_LONGITUDE = 0.0
    OFFSET = SCALING_FACTOR = 0.0
    SAMPLE_TYPE = UNIT = TARGET_NAME = RadiusUM = Message = ""
    start_up = True

    bpy.types.Scene.fpath = bpy.props.StringProperty(
        name="Import File ",
        description="Select your img file",
        subtype="FILE_PATH",
        default="",
        update=update_fpath,
    )

    # --- Transform properties (always available) ---
    bpy.types.Scene.TransformScale = bpy.props.FloatProperty(
        name="Scale",
        description="Uniform scale applied to the imported object",
        min=0.0001,
        max=10000.0,
        default=1.0,
        precision=4,
    )
    bpy.types.Scene.HeightExaggeration = bpy.props.FloatProperty(
        name="Height Exaggeration",
        description="Multiplier applied to DEM pixel values to exaggerate terrain relief (1 = no exaggeration)",
        min=0.0,
        max=1000.0,
        default=1.0,
        precision=3,
    )
    bpy.types.Scene.UpAxis = bpy.props.EnumProperty(
        name="Up Axis",
        description="Which axis points up in the imported mesh",
        items=[
            ('X',  "X",  ""),
            ('Y',  "Y",  ""),
            ('Z',  "Z",  ""),
        ],
        default='Z',
    )
    bpy.types.Scene.ForwardAxis = bpy.props.EnumProperty(
        name="Forward Axis",
        description="Which axis points forward in the imported mesh",
        items=[
            ('X',         "X",  ""),
            ('Y',         "Y",  ""),
            ('Z',         "Z",  ""),
            ('NEGATIVE_X', "-X", ""),
            ('NEGATIVE_Y', "-Y", ""),
            ('NEGATIVE_Z', "-Z", ""),
        ],
        default='Y',
    )


def clear_properties():
    if bpy.context.scene is None:
        return

    global MAXIMUM_LATITUDE, MINIMUM_LATITUDE
    global WESTERNMOST_LONGITUDE, EASTERNMOST_LONGITUDE
    global LINES, LINE_SAMPLES, SAMPLE_BITS, MAP_RESOLUTION
    global OFFSET, SCALING_FACTOR
    global SAMPLE_TYPE, UNIT, TARGET_NAME, RadiusUM, Message
    global start_up

    LINES = LINE_SAMPLES = SAMPLE_BITS = MAP_RESOLUTION = 0
    MAXIMUM_LATITUDE = MINIMUM_LATITUDE = 0.0
    WESTERNMOST_LONGITUDE = EASTERNMOST_LONGITUDE = 0.0
    OFFSET = SCALING_FACTOR = 0.0
    SAMPLE_TYPE = UNIT = TARGET_NAME = RadiusUM = Message = ""
    start_up = True

    props = ["FromLat", "ToLat", "FromLong", "ToLong", "Scale", "Magnify", "fpath",
             "TransformScale", "HeightExaggeration", "UpAxis", "ForwardAxis"]
    for p in props:
        if p in bpy.types.Scene.bl_rna.properties:
            exec("del bpy.types.Scene." + p)
        if p in bpy.context.scene:
            del bpy.context.scene[p]

    bpy.types.Scene.fpath = bpy.props.StringProperty(
        name="Import File ",
        description="Select your img file",
        subtype="FILE_PATH",
        default="",
        update=update_fpath,
    )

    # Re-register transform properties after reset
    bpy.types.Scene.TransformScale = bpy.props.FloatProperty(
        name="Scale",
        description="Uniform scale applied to the imported object",
        min=0.0001,
        max=10000.0,
        default=1.0,
        precision=4,
    )
    bpy.types.Scene.HeightExaggeration = bpy.props.FloatProperty(
        name="Height Exaggeration",
        description="Multiplier applied to DEM pixel values to exaggerate terrain relief (1 = no exaggeration)",
        min=0.0,
        max=10.0,
        default=1.0,
        precision=3,
    )
    bpy.types.Scene.UpAxis = bpy.props.EnumProperty(
        name="Up Axis",
        description="Which axis points up in the imported mesh",
        items=[
            ('X',  "X",  ""),
            ('Y',  "Y",  ""),
            ('Z',  "Z",  ""),
        ],
        default='Z',
    )
    bpy.types.Scene.ForwardAxis = bpy.props.EnumProperty(
        name="Forward Axis",
        description="Which axis points forward in the imported mesh",
        items=[
            ('X',         "X",  ""),
            ('Y',         "Y",  ""),
            ('Z',         "Z",  ""),
            ('NEGATIVE_X', "-X", ""),
            ('NEGATIVE_Y', "-Y", ""),
            ('NEGATIVE_Z', "-Z", ""),
        ],
        default='Y',
    )
_classes = (
    Import,
    Reset,
    Img_Importer,
)


def register():
    initialize()
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    clear_properties()


if __name__ == "__main__":
    register()
