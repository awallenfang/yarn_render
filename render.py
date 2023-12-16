## notes
# emitter pdf direction si
# si' from get_out_pos on si
# emitter pdf si'


import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
mi.set_variant('llvm_ad_rgb')

scene = mi.load_file("scenes/tea_cozy_scene.xml")
# scene = mi.load_dict(mi.cornell_box())


image = mi.render(scene, spp=128)

plt.axis('off')
plt.imshow(image ** (1.0 / 2.2))
plt.show()