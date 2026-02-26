for Blender 5 ( should ? run on Blender 4.5 exr/lts)

----- Import_cmod_mesh_parallel.py -----
imports ONLY the ASCII format cmod

for the CMOD import plugin the cmod MUST be in ASCII format
use " cmodfix -a input.bin.cmod output.ascii.cmod " on the commandline

use copy/past to add the location of the textures folder IF!!! they are not in the same folder as the cmod
full system path is needed
like the celestia addon file structure
```

.
└── addon
    ├── models
    │   └── mesh.ascii.cmod
    └── textures
        └── medres
            ├── image1.png
            └── image2.png

"/home/user/celestia/extras/addon/textures/medres"

```


----- Export_cmod_mesh_parallel.py -----

if the textures are packed into the blend file use
File / External Data / Unpack Resources

or move them into the file structure above to use "cmodview-qt6"





----- io_import_LRO_Lola_MGS_Mola_img.py -----
it installs to "3D Viewport > Sidebar > LRO/MGS" ( hit the "n" button)

-- converts a 2d simple cylindrical DEM into a 3d spherical mesh --

imports NASA/JPL/USGS/... planet/moon PDS3 DEM's ( height maps)
for example :

https://pds-geosciences.wustl.edu/missions/mgs/megdr.html

points to the PDS3 archive

https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/

megt90n000cb.img
megt90n000cb.lbl

