## notes
# emitter pdf direction si
# si' from get_out_pos on si
# emitter pdf si'


import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from mitsuba_modules.bundle_bsdf import BundleBSDF

mi.set_variant('llvm_spectral')
from mitsuba_modules.bundle_integrator import BundleIntegrator

# mi.register_integrator('bundle', lambda props: BundleIntegrator(props))
mi.register_bsdf('bundle_bsdf', lambda props: BundleBSDF(props))
# print("Successfully registered integrator")

# scene = mi.load_file("scenes/tea_cozy_scene.xml")
# scene = mi.load_dict(mi.cornell_box())
scene = mi.load_file("inside_test.xml")
# scene = mi.load_file("intersect_scenes/flame_ribbing_pattern_intersect_scene.xml")
image = mi.render(scene)

plt.axis('off')
plt.imshow(image ** (1.0 / 2.2))
plt.show()