# Progressive-design frames for the page's generation demo: the same complex,
# palette and camera as Fig. 1, with a growing share of it designed. Run with
#   /Applications/PyMOL.app/Contents/MacOS/PyMOL -cq docs/render_frames.py
import os, math, random
from pymol import cmd
from PIL import Image

# Both are relative to the directory this is run from, which the line above
# makes the repository root. Not __file__: under PyMOL's own `run` that
# resolves inside the pymol package, not next to this script. FIG1_DIR holds
# the structures the figure is built from and is not part of this repository.
HERE = os.environ.get("FIG1_DIR", "fig1_pymol")
OUT  = os.environ.get("FRAME_DIR", "docs/static/images")
W, H, NF = 760, 546, 13
TW, PAD = 620, 6

# Fig. 1's palette and its frozen camera, so the panel and the page show the
# same picture of 1JID: human SRP19 on helix 6 of SRP RNA, two magnesium ions.
COLORS = {"designed": [0.478, 0.573, 0.831],
          "fixed":    [0.831, 0.847, 0.867],
          "cofactor": [0.729, 0.475, 0.353]}
VIEW = (0.3772266209125519, 0.06398552656173706, 0.9239079356193542,
        0.35775429010391235, 0.9101041555404663, -0.2090986967086792,
        -0.8542317152023315, 0.40940964221954346, 0.3204244375228882,
        0.0, 0.0, -203.8352508544922,
        16.86206817626953, 14.752896308898926, 33.15565872192383,
        160.70526123046875, 246.96524047851562, -20.0)

cmd.reinitialize()
for n, c in COLORS.items(): cmd.set_color(n, c)
cmd.bg_color("white")
cmd.set("ray_opaque_background", 0)
cmd.set("orthoscopic", 1); cmd.set("antialias", 2); cmd.set("valence", 0)
cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_color", "grey10")
cmd.set("ray_trace_gain", 0.10); cmd.set("ray_trace_disco_factor", 1)
cmd.set("specular", 0); cmd.set("spec_reflect", 0)
cmd.set("ambient", 1.00); cmd.set("direct", 0.00); cmd.set("reflect", 0.00)
cmd.set("power", 1.0); cmd.set("cartoon_fancy_helices", 0)
cmd.set("ray_shadows", 0); cmd.set("depth_cue", 0); cmd.set("fog", 0)
cmd.set("cartoon_smooth_loops", 1); cmd.set("cartoon_loop_radius", 0.30)
cmd.set("cartoon_flat_sheets", 1); cmd.set("cartoon_transparency", 0)
cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1)
cmd.set("cartoon_nucleic_acid_mode", 4); cmd.set("cartoon_ring_transparency", 0.0)
cmd.set("cartoon_ladder_mode", 1); cmd.set("sphere_scale", 0.60)

cmd.load(os.path.join(HERE, "1jid_complex.pdb"), "cx")
cmd.remove("solvent"); cmd.hide("everything")
cmd.select("prot", "cx and polymer.protein")
cmd.select("na", "cx and polymer.nucleic")
cmd.select("ions", "cx and inorganic")
cmd.show("cartoon", "prot or na"); cmd.show("spheres", "ions")
cmd.dss("prot"); cmd.deselect()
cmd.set_view(VIEW)

# Watson-Crick partners from the hydrogen-bonded ring nitrogens, purine N1 to
# pyrimidine N3 under 3.4 A, nearest candidate wins. The RNA here is one chain
# folded back on itself, so these pairs are inside it.
PUR = {"A", "G", "DA", "DG"}
ring = {}
for a in cmd.get_model("na and name N1+N3").atom:
    if a.name == ("N1" if a.resn in PUR else "N3"):
        ring[(a.chain, a.resi)] = (a.resn, a.coord)
pairs, taken = {}, set()
for i in sorted(ring):
    if i in taken: continue
    best = None
    for j in ring:
        if j in taken or j == i: continue
        if i[0] == j[0] and abs(int(i[1]) - int(j[1])) < 3: continue
        if (ring[i][0] in PUR) == (ring[j][0] in PUR): continue
        d = math.dist(ring[i][1], ring[j][1])
        if d < 3.4 and (best is None or d < best[1]): best = (j, d)
    if best: pairs[i] = best[0]; taken.update((i, best[0]))

# A base pair is one unit, so the rungs of the helix light up together, the way
# a base and its partner are decoded against each other.
units  = [[(a.chain, a.resi)] for a in cmd.get_model("prot and name CA").atom]
units += [[k, pairs[k]] for k in pairs]
units += [[k] for k in ring if k not in taken]
random.Random(7).shuffle(units)
order = [r for u in units for r in u]
log = open("/tmp/zf_render.log", "w")
print("%d protein, %d nucleic, %d pairs, %d ions, %d designable"
      % (cmd.count_atoms("prot and name CA"), len(ring), len(pairs),
         cmd.count_atoms("ions"), len(order)), file=log)
print("pairs: " + ", ".join("%s%s-%s%s" % (k[0], k[1], v[0], v[1])
                            for k, v in sorted(pairs.items())), file=log)

raw = []
for f in range(NF):
    k = round(len(order) * f / (NF - 1))
    cmd.color("fixed", "prot or na"); cmd.color("cofactor", "ions")
    if k:
        cmd.color("designed", "(%s)" % " or ".join("(chain %s and resi %s)" % cr for cr in order[:k]))
    p = "/tmp/zf_raw_%02d.png" % f
    cmd.ray(W, H); cmd.png(p, dpi=300); raw.append(p)

# One crop for every frame, then 64 colours: flat fills lose nothing and the
# thirteen frames come to about half a megabyte.
ims = [Image.open(p).convert("RGBA") for p in raw]
bb = None
for im in ims:
    b = im.split()[3].getbbox()
    bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
bb = (bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD)
th = round((bb[3] - bb[1]) * TW / (bb[2] - bb[0]))
for f, im in enumerate(ims):
    im = im.crop(bb).resize((TW, th), Image.LANCZOS)
    q = im.convert("RGB").quantize(colors=64, method=Image.FASTOCTREE).convert("RGBA")
    q.putalpha(im.split()[3])
    q.quantize(colors=64, method=Image.FASTOCTREE).save(os.path.join(OUT, "zf%02d.png" % f), optimize=True)
print("%d frames at %dx%d -> %s" % (NF, TW, th, OUT), file=log)
log.close()
cmd.quit()
