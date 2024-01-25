import struct
from os.path import exists

import numpy as np

up_vectors = {
    1: "0.0, 1.0, 0.0",
    2: "0.0, 0.0, 1.0",
}
INT_SIZE = 4
FLOAT_SIZE = 4

PLY_AMT = 3
WINDING_SPACING = 0.1
THREAD_RADIUS = 0.2

file_name = "flame_ribbing_pattern.bcc"
file = open(f'patterns/{file_name}', "rb")
data = file.read()
file.close()

# Check if the files were converted already
converted_already = exists(f'linear_curves/{file_name.split(".")[0]}_0.txt')
        
file_name = file_name.split(".")[0]

header_bytes = data[0:64]
print(len(header_bytes))
# Read in the header data using the struct module
# The structure is the following
# 1. 3 ASCII letters for magic numbers
# 2. 1 hexadecimal number for the precision of the data
# 3. 2 ASCII letters for the type of curve that's used
# 4. 1 byte for the number of dimensions
# 5. 1 byte for the dimension that is the up vector
# 6. 64 bit unsigned integer containing the total amount of curves
# 7. 64 bit unsigned integer containing the total amount of control points
# 8. 39 ASCII letters for additional data
header = struct.unpack("3s1s2s2B2Q40s", header_bytes)

magic_numbers = header[0].decode("utf-8")
precision = header[1]
curve_type = header[2].decode("utf-8")
dimensions = header[3]
up_vector = header[4]
total_curves = header[5]
total_control_points = header[6]
additional_data = header[7].decode("utf-8")

print("Magic numbers: " + magic_numbers)
print("Precision: " + precision.hex())
print("Curve type: " + curve_type)
print("Dimensions: " + str(dimensions))
print("Up vector: " + str(up_vector))
print("Total curves: " + str(total_curves))
print("Total control points: " + str(total_control_points))
print("Additional data: " + additional_data)

data_bytes = data[64:]


# Linear output

data_idx = 0
points_left = 0
curves_arr = []

curve_data = []
interp_curve_data = []

curve_plies = [[[] for _ in range(PLY_AMT)] for _ in range(total_curves)]

while data_idx < len(data_bytes):
    # If there is a new curve add a new list to the curves array
    if points_left == 0:
        curves_arr.append([])
        curve_data.append([])
        interp_curve_data.append([])
        points_left = abs(int.from_bytes(data_bytes[data_idx:data_idx+INT_SIZE], byteorder='little', signed=True))
        # print(f'Points in curve: {points_left}')
        data_idx += INT_SIZE
        continue

    # Grab the bytes and write them to the last curve array
    x_bytes = data_bytes[data_idx:data_idx+FLOAT_SIZE]
    y_bytes = data_bytes[data_idx+FLOAT_SIZE:data_idx+FLOAT_SIZE*2]
    z_bytes = data_bytes[data_idx+2*FLOAT_SIZE:data_idx+FLOAT_SIZE*3]

    x = struct.unpack('f', x_bytes)[0]
    y = struct.unpack('f', y_bytes)[0]
    z = struct.unpack('f', z_bytes)[0]

    curves_arr[-1].append(f'{x} {y} {z} 0.2\n')
    curve_data[-1].append((x,y,z))

    # If there are at least two points in the current curve_data
    if len(curve_data[-1]) > 1:
        # Get the two points and interpolate them in equal stepsizes
        # Then write the new point to interp_curve_data
        p1 = curve_data[-1][-2]
        p2 = curve_data[-1][-1]
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        spacing = WINDING_SPACING
        steps = int(distance / spacing)
        if steps == 0:
            steps = 1
        for i in range(steps):
            interp_curve_data[-1].append((p1[0] + (p2[0] - p1[0]) * i / steps, p1[1] + (p2[1] - p1[1]) * i / steps, p1[2] + (p2[2] - p1[2]) * i / steps))

    data_idx += FLOAT_SIZE*3
    points_left -= 1

# Iterate through the interp_curve_data and write the points to the curve_plies
for curve_idx in range(len(interp_curve_data)):
    for ply_idx in range(PLY_AMT):
        phase = 2*np.pi/PLY_AMT * ply_idx
        for point_idx in range(len(interp_curve_data[curve_idx])):
            # The ply_point is the base point plus the offset
            # The offset is the winding, rotated by the current direction of the curve
            direction = np.array(interp_curve_data[curve_idx][point_idx]) - np.array(interp_curve_data[curve_idx][point_idx-1])
            offset = np.array([np.cos(point_idx * WINDING_SPACING + phase), np.sin(point_idx * WINDING_SPACING + phase), 0]) * THREAD_RADIUS
            # Rotate offset into the direction
            
            ply_point = interp_curve_data[curve_idx][point_idx] 
            curve_plies[curve_idx][ply_idx].append(interp_curve_data[curve_idx][point_idx])


curve_amt = len(curves_arr)
print(curve_amt)
for i in range(curve_amt):
    out_string = "".join(curves_arr[i])
    with open(f'linear_curves/{file_name}_{i}.txt', "+w") as file:
        file.write(out_string)

# Convert the linear points into mitsuba3 cylinders
cylinders = ""
# Flatten the curves array
curves_arr = np.array(curves_arr)
curves_arr = curves_arr.flatten()
for i in range(len(curves_arr)-1):
    p0 = ", ".join(curves_arr[i].split(" ")[0:3])
    p1 = " ".join(curves_arr[i+1].split(" ")[0:3])

    cylinders += f'\t<shape type="cylinder">\n\t\t<float name="radius" value="0.1"/>\n\t\t<vector name="p0" value="{p0}"/>\n\t\t<vector name="p1" value="{p1}"/>\n\t\t<bsdf type="twosided"><bsdf type="diffuse"><rgb name="reflectance" value="0.2, 0.25, 0.7"/></bsdf></bsdf>\n\t</shape>\n'

        

# Write the mitsuba data to the curve_preset.xml file to generate the input

output = ""

for i in range(curve_amt):
    output += f'\t<shape type="linearcurve">\n\t\t<transform name="to_world">\n\t\t\t<translate x="1" y="0" z="0"/>\n\t\t\t<scale value="2"/>\n\t\t</transform>\n\t\t<string name="filename" value="linear_curves/{file_name}_{i}.txt"/>\n\t</shape>\n'





with open(f'scenes/{file_name}_scene.xml', "+w") as scene_file:
    with open("curve_preset.xml", "r") as file:
        data = file.read()
        data = data.replace("<!-- UP VECTOR -->", up_vectors[up_vector])
        data = data.replace("<!-- CURVES HERE -->", output)
        scene_file.write(data)        

with open(f'intersect_scenes/{file_name}_intersect_scene.xml', "+w") as inter_file:
    with open("curve_preset.xml", "r") as file:
        data = file.read()
        data = data.replace("<!-- UP VECTOR -->", up_vectors[up_vector])
        data = data.replace("<!-- CURVES HERE -->", cylinders)
        inter_file.write(data)