from PIL import ImageChops, ImageFilter, Image
import numpy as np
import cv2


def multiply_nn_mnn(g, rgb):
    rgb[:,:,0] = rgb[:,:,0] * g
    rgb[:,:,1] = rgb[:,:,1] * g
    rgb[:,:,2] = rgb[:,:,2] * g

    return rgb

def cv_multiresolution_blend(gm, la, lb) -> list:

    gm = [x // 255 for x in gm]
    blended = []
    for i in range(len(gm)):
        #gmi , lbi = cv2_same_size(gm[i], lb[i])
        gmi,lbi,lai=gm[i], lb[i], la[i]
        la_width = lai.shape[0]
        la_height = lai.shape[1]
        l_width = lbi.shape[0]
        l_height = lbi.shape[1]
        d_width = l_width - la_width
        d_height = l_height - la_height
        if d_width != 0:
            if d_width>0:
                lai = cv2.copyMakeBorder(lai, d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                lbi = cv2.copyMakeBorder(lbi, -d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
        if d_height != 0:
            if d_height>0:
                lai = cv2.copyMakeBorder(lai, 0, 0, d_height, 0, cv2.BORDER_REPLICATE)
            else:
                lbi = cv2.copyMakeBorder(lbi, 0, 0, -d_height, 0, cv2.BORDER_REPLICATE)

        g_width=gmi.shape[0]
        g_height=gmi.shape[1]
        l_width=lbi.shape[0]
        l_height=lbi.shape[1]
        d_width=l_width-g_width
        d_height=l_height-g_height
        if d_width != 0:
            if d_width>0:
                gmi = cv2.copyMakeBorder(gmi, d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                lai = cv2.copyMakeBorder(lai, -d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
                lbi = cv2.copyMakeBorder(lbi, -d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
        if d_height != 0:
            if d_height>0:
                gmi = cv2.copyMakeBorder(gmi, 0, 0, d_height, 0, cv2.BORDER_REPLICATE)
            else:
                lai = cv2.copyMakeBorder(lai, 0, 0, -d_height, 0, cv2.BORDER_REPLICATE)
                lbi = cv2.copyMakeBorder(lbi, 0, 0, -d_height, 0, cv2.BORDER_REPLICATE)


        bi = multiply_nn_mnn(gmi, lbi) + multiply_nn_mnn((1-gmi), lai)
        bi = bi.astype(np.uint8)
        blended.append(bi)
    return blended

def cv_reconstruct_laplacian(blended_pyramid):

    scale = len(blended_pyramid)
    up = blended_pyramid[-1] # start with the tip, this is would the smallest scale image, while the rest of the pyramid would contain blended laplacians
    for i in range(scale-1, 0, -1):
        next = blended_pyramid[i-1].copy()
        up = cv2.pyrUp(up)


        #up, next = cv2_same_size(up, next) #sometimes the width/height can be off by a few pixels due to `cv2.pyrUp`

        g_width=up.shape[0]
        g_height=up.shape[1]
        l_width=next.shape[0]
        l_height=next.shape[1]
        d_width=l_width-g_width
        d_height=l_height-g_height
        if d_width != 0:
            if d_width>0:
                up = cv2.copyMakeBorder(up, d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                next = cv2.copyMakeBorder(next, -d_width, 0, 0, 0, cv2.BORDER_REPLICATE)

        if d_height != 0:
            if d_height>0:
                up = cv2.copyMakeBorder(up, 0, 0, d_height, 0, cv2.BORDER_REPLICATE)
            else:
                next = cv2.copyMakeBorder(next, 0, 0, -d_height, 0, cv2.BORDER_REPLICATE)
        up = cv2.add(next, up)
        print(f"{next.shape} - {up.shape}")

    return up

def smoothing_kernel() -> ImageFilter.Kernel:
    kernel_size = (3,3)
    # kernels are defined row-wise
    kernel = [\
        1, 2, 1,\
        2, 4, 2,\
        1, 2, 1]
    return ImageFilter.Kernel(kernel_size, kernel)

def cv_pyramid(A, scale):

    gp = [A]
    for i in range(1, scale):
        A = cv2.pyrDown(A)
        gp.append(A)
    return gp

def cv_laplacian(gp, scale) -> list:

    lp = [gp[-1].copy()]
    for i in reversed(range(scale-1)):
        gExp = cv2.pyrUp(gp[i+1].copy())
        gpi = gp[i].copy()

        g_width=gpi.shape[0]
        g_height=gpi.shape[1]
        l_width=gExp.shape[0]
        l_height=gExp.shape[1]
        d_width=l_width-g_width
        d_height=l_height-g_height
        if d_width != 0:
            if d_width>0:
                gpi = cv2.copyMakeBorder(gpi, d_width, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                gExp = cv2.copyMakeBorder(gExp, -d_width, 0, 0, 0, cv2.BORDER_REPLICATE)

        if d_height != 0:
            if d_height>0:
                gpi = cv2.copyMakeBorder(gpi, 0, 0, d_height, 0, cv2.BORDER_REPLICATE)
            else:
                gExp = cv2.copyMakeBorder(gExp, 0, 0, -d_height, 0, cv2.BORDER_REPLICATE)
        #gpi, gExp = cv2_same_size(gpi, gExp)
        li = cv2.subtract(gpi, gExp)
        lp.insert(0, li)
    return lp