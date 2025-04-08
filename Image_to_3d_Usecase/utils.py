#utils.py: utility stuff like reading and writing.

import numpy as np
import cv2

#load img from a folder images/. MUST: images are named "0000.jpg", "0001.jpg", etc.
def loadImages(folderName, numberOfImages):
    imagesLoaded = []
    for i in range(numberOfImages):
        fileName = f"{folderName}/{i:04d}.jpg"
        image = cv2.imread(fileName)
        imagesLoaded.append(image)
    return imagesLoaded

#read camera intrinsic matrix from K.txt file
def readCalibrationMatrix(filePath="K.txt"):
    return np.loadtxt(filePath)

#write 3D points to PLY file
#A basic .ply file format example with points and colors is acceptable.
def writePLY(outputFileName, points ):
    with open(outputFileName, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n" )
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        colors = np.tile(np.array( [204, 204, 255], dtype=np.uint8), (len(points), 1))
        for coordinate, color in zip(points, colors):
            file.write(f"{coordinate[0]} {coordinate[1]} {coordinate[2]} "f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
