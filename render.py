## notes
# emitter pdf direction si
# si' from get_out_pos on si
# emitter pdf si'


import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from mitsuba_modules.bundle_bsdf import BundleBSDF
import numpy as np
import colour


# mi.set_variant('llvm_mono')

# images = np.zeros((1000, 1000, 3, 11))

# mi.register_bsdf('bundle_bsdf', lambda props: BundleBSDF(props))
# # print("Successfully registered integrator")

# # scene = mi.load_file("scenes/tea_cozy_scene.xml")
# # scene = mi.load_dict(mi.cornell_box())

# # Load the scene file into a string and replace WAVELENGTH with the wavelength
# scene = ""
# with open("inside_test.xml", "r") as file:
#     scene = file.read()

# for w in range(400, 710, 300//10):
#     print("Working on wavelength: ", w)
#     scene = ""
#     with open("inside_test.xml", "r") as file:
#         scene = file.read()
#     scene = scene.replace("WAVELENGTH", str(float(w)))
#     scene_obj = mi.load_string(scene)
#     image = mi.render(scene_obj).numpy()
#     images[:, :, 0, (w - 400) // (300//10)] = image.reshape(1000,1000)
#     images[:, :, 1, (w - 400) // (300//10)] = image.reshape(1000,1000)
#     images[:, :, 2, (w - 400) // (300//10)] = image.reshape(1000,1000)

#     # Save the images to a folder for further use
#     np.save("images/inside_test_" + str(w), image)

# scene = scene.replace("WAVELENGTH", "400.")


# scene_obj = mi.load_string(scene)
# # scene = mi.load_file("inside_test.xml")
# # scene = mi.load_file("intersect_scenes/flame_ribbing_pattern_intersect_scene.xml")
# image = mi.render(scene_obj)

# plt.axis('off')
# plt.imshow(images[:,:,2] ** (1.0 / 2.2))
# plt.show()

output = np.zeros((1000, 1000, 3))

# Convert the images to XYZ and then to sRGB
for w in range(400, 710, 300//10):
    img = np.load("images/inside_test_" + str(w) + ".npy")
    rgb = colour.XYZ_to_RGB(colour.wavelength_to_XYZ(float(w)), colourspace=colour.models.RGB_COLOURSPACE_sRGB)
    print(rgb)
    # images[:, :, 0, (w - 400) // (300//10)] *= rgb[0]
    # images[:, :, 1, (w - 400) // (300//10)] *= rgb[1]
    # images[:, :, 2, (w - 400) // (300//10)] *= rgb[2]

    output += img * rgb


# # Save the result to an array
# xyz = colour.wavelength_to_XYZ(images)
# rgb_images = colour.XYZ_to_sRGB(xyz)
# Additively combine the images
# final_image = np.sum(output, axis=3) / 8.
# plt.imshow(output)
# plt.show()

# output = np.zeros((1000, 1000, 3))

# for i in range(11):
#     output += output[:,:,:,i]

output /= 5.
print(np.min(output), np.max(output))
plt.imshow(output)
plt.show()