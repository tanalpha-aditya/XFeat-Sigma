from pyramid import cv_laplacian, cv_pyramid, cv_multiresolution_blend, cv_reconstruct_laplacian


def pyramid_blend(dst_img,src_img,mask,scale):

    gpA = cv_pyramid(dst_img.copy(), scale)
    gpB = cv_pyramid(src_img.copy(), scale)
    gpM = cv_pyramid(mask.copy(), scale)
    lpA = cv_laplacian(gpA, scale)
    lpB = cv_laplacian(gpB, scale)
    blended_pyramid = cv_multiresolution_blend(gpM, lpA, lpB)
    blended_image = cv_reconstruct_laplacian(blended_pyramid)
    return blended_image
