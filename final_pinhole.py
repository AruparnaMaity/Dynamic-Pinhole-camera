import numpy as np
import matplotlib.pyplot as plt
import cv2


# Distance of pin-hole from object
# Distance of screen from pin-hole
L = 77
d = 100

# Co-ordinates of pin-hole and the screen
p_x = 0
p_y = 0
p_origin = [0, 0, L]
p = [p_x, p_y, L]
s = [0, 0, (L + d)]


obj_path = "sample_signature.jpg"
img_BGR = cv2.imread("{}".format(obj_path))
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_RGB_0 = img_RGB[::-1, :, 0]

## Plotting the object
#plt.axis([-w_obj, w_obj, -h_obj, h_obj])
#plt.imshow(img_RGB_0, cmap = "gray")
#plt.show()

img_RGB_0_flatten = img_RGB_0.flatten()

a = list(range(0, img_RGB_0.shape[0]))
b = list(range(0, img_RGB_0.shape[1]))

l = []
for i in range(len(a)):
    for j in range(len(b)):
        l.append([a[i]] + [b[j]])
 
V = np.array(l)
V[:, 0] = V[:, 0] + p_x
V[:, 1] = V[:, 1] + p_y
    
# Building the dictionary of object coordinates and the corresponding pixel values
dict1 = {}
for i in range(len(l)):
    dict1[str(l[i])] = img_RGB_0_flatten[i]


# Computing theta (the angle between the ray and the axis of pin-hole camera)
theta = []
for i in range(len(V)):
    s = 0
    for j in range(len(V[i])):
        s += np.square(V[i][j])
    theta.append(np.degrees(np.arctan([np.sqrt(s) / L])))
    
# Computing phi (the angle between the the object point vector and x-axis)
phi = []
for i in range(len(V)):
    s = 0
    for j in range(len(V[i])):
        s += np.square(V[i][j])
    if V[i][0] <= 0 and V[i][1] < 0:
        phi.append(270 - np.degrees(np.arcsin([np.absolute(V[i][0]) / np.sqrt(s)])))
    elif V[i][0] < 0 and V[i][1] >= 0:
        phi.append(90 + np.degrees(np.arcsin([np.absolute(V[i][0]) / np.sqrt(s)])))
    elif V[i][0] >= 0 and V[i][1] > 0:
        phi.append(np.degrees(np.arcsin([np.absolute(V[i][1]) / np.sqrt(s)])))
    elif V[i][0] > 0 and V[i][1] <= 0:
        phi.append(360 - np.degrees(np.arcsin([np.absolute(V[i][1]) / np.sqrt(s)])))
    elif V[i][0] == 0 and V[i][1] == 0:
        phi.append(0)
len(phi)

'''
e = p - V

# Calculating distance of corresponding image points on screen from the pin-hole
l = []
for i in range(len(e)):
    l.append((d+L) / (L/np.linalg.norm(e[i])))
'''    

# Distance of image pixels from centre
m = []
for i in range(len(V)):
    s = 0
    for j in range(len(V[i])):
        s += np.square(V[i][j])
    m.append(np.sqrt(s) * (d/L))
 
### Image pixel coordinates (given by array U)    
U = np.zeros((V.shape[0], V.shape[1]))
img_obj = []

for j in range(len(V)):
    if V[j][0] <= 0 and V[j][1] < 0:
        J = m[j] * np.cos(np.radians(phi[j])), m[j] * np.sin(np.radians(phi[j])), d+L
#        print(J)
#        for x,y,z in zip(J[0], J[1], J[2]):
#            U[j] = x,y,z
        U[j] = J[0], J[1]#, J[2]
#        print(V[j],U[j])
        img_obj.append((V[j],U[j]))       
    elif V[j][0] < 0 and V[j][1] >= 0:
        J = m[j] * np.cos(np.radians(phi[j])), m[j] * np.sin(np.radians(phi[j])), d+L
#        print(J)
#        for x,y,z in zip(J[0], J[1], J[2]):
#            U[j] = x,y,z
        U[j] = J[0], J[1]#, J[2]
#        print(V[j],U[j])
        img_obj.append((V[j],U[j]))
    elif V[j][0] >= 0 and V[j][1] > 0:
        J = m[j] * np.cos(np.radians(phi[j])), m[j] * np.sin(np.radians(phi[j])), d+L
#        print(J)
#        for x,y,z in zip(J[0], J[1], J[2]):
#            U[j] = x,y,z
        U[j] = J[0], J[1]#, J[2]
#        print(V[j],U[j])
        img_obj.append((V[j],U[j]))
    elif V[j][0] > 0 and V[j][1] <= 0:
        J = m[j] * np.cos(np.radians(phi[j])), m[j] * np.sin(np.radians(phi[j])), d+L
#        print(J)
#        for x,y,z in zip(J[0], J[1], J[2]):
#            U[j] = x,y,z
        U[j] = J[0], J[1]#, J[2]
#        print(V[j],U[j])
        img_obj.append((V[j],U[j]))
    elif V[j][0] == 0 and V[j][1] == 0:
#        print(V[j],U[j])
        img_obj.append((V[j],U[j]))

# Dictionary of coordinates: key = image coordinates, value = object coordinates
dict2 = {}
for i in range(len(img_obj)):
    img_obj_il1 = [j for j in img_obj[i][0]]
    img_obj_il2 = [j for j in img_obj[i][1]]
#    print(img_obj_il1 + img_obj_il2)
#    img_obj_il = [j for j in img_obj[0]]
    dict2[str(img_obj_il2)] = img_obj_il1


'''
# Writing the dictionaries to respective files
with open("I:/Data Science/ISI/Pinhole/Python new/dict2.txt",'w') as data:
    data.write(str(dict2))
with open("I:/Data Science/ISI/Pinhole/Python new/dict1.txt",'w') as data:
    data.write(str(dict1))
'''


# Transfering the pixel values from the object to the image
c1 = [dict1["{}".format(dict2["{}".format(i)])] for i in dict2]

w_screen = np.max(U - p_x)
h_screen = np.max(U - p_y)

w_obj = np.max(img_RGB_0_flatten)
h_obj = np.max(img_RGB_0_flatten)

horizontal_size = max(w_screen, w_obj)
vertical_size = max(h_screen, h_obj)

# Plotting the object and the image
plt.axis([-horizontal_size, horizontal_size, -vertical_size, vertical_size])
plt.imshow(img_RGB_0, cmap = "gray")
plt.scatter(-U[:,1] + p_x, -U[:,0] + p_y,  c = c1, cmap = "gray")
plt.show()
