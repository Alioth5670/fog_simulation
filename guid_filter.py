import numpy as np
import cv2


def _gf_color(I, p, r, eps,):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    if isinstance(r,int):
        r = (r,r)

    h, w = p.shape[:2]
    N = cv2.boxFilter(np.ones((h, w)), -1, r)

    mI_r = cv2.boxFilter(I[:,:,0], -1, r) / N
    mI_g = cv2.boxFilter(I[:,:,1], -1, r) / N
    mI_b = cv2.boxFilter(I[:,:,2], -1, r) / N

    mP = cv2.boxFilter(p, -1, r) / N

    # mean of I * p
    mIp_r = cv2.boxFilter(I[:,:,0]*p, -1, r) / N
    mIp_g = cv2.boxFilter(I[:,:,1]*p, -1, r) / N
    mIp_b = cv2.boxFilter(I[:,:,2]*p, -1, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = cv2.boxFilter(I[:,:,0] * I[:,:,0], -1, r) / N - mI_r * mI_r
    var_I_rg = cv2.boxFilter(I[:,:,0] * I[:,:,1], -1, r) / N - mI_r * mI_g
    var_I_rb = cv2.boxFilter(I[:,:,0] * I[:,:,2], -1, r) / N - mI_r * mI_b

    var_I_gg = cv2.boxFilter(I[:,:,1] * I[:,:,1], -1, r) / N - mI_g * mI_g
    var_I_gb = cv2.boxFilter(I[:,:,1] * I[:,:,2], -1, r) / N - mI_g * mI_b

    var_I_bb = cv2.boxFilter(I[:,:,2] * I[:,:,2], -1, r) / N - mI_b * mI_b

#######################
# 计算速度可以优化(calculate speed can be optimised by unfold for-loop)

# (1080*1920) time: 32.599939823150635
    # a = np.zeros((h, w, 3))
    # for i in range(h):
    #     for j in range(w):
    #         sig = np.array([
    #             [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
    #             [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
    #             [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
    #         ])
    #         covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
    #         a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

###############################
# 老子优化完了hhhh
# (1080*1920) time: 1.556933879852295
    # 将一维的 covIp 向量转换为三维，以便与 sig 矩阵进行匹配操作
    # convert 1-d covIp to 3-d for matching with sig matrix
    covIp_3d = np.stack((covIp_r, covIp_g, covIp_b), axis=-1)

    eps_eye = eps * np.eye(3)

    var_I_transposed = np.stack((
        var_I_rr, var_I_rg, var_I_rb,
        var_I_rg, var_I_gg, var_I_gb,
        var_I_rb, var_I_gb, var_I_bb
    ), axis=-1).reshape((h,w,3,3))
    
    # 批量计算逆矩阵并解线性方程组
    # calculate inverse matrix and sove linear equations
    a = np.linalg.solve(var_I_transposed + eps_eye[None, None,:,:], covIp_3d)
######################################
    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = cv2.boxFilter(a, -1, r) / N[...,np.newaxis]
    meanB = cv2.boxFilter(b, -1, r) / N

    q = np.sum(meanA * I, axis=2) + meanB

    return q


def _gf_gray(I, P, r, eps):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
    """

    (rows, cols) = I.shape

    N = cv2.boxFilter(np.ones([rows, cols]), r)

    meanI = cv2.boxFilter(I, -1, r) / N
    meanP = cv2.boxFilter(P, -1, r) / N
    corrI = cv2.boxFilter(I * I, -1, r) / N
    corrIp = cv2.boxFilter(I * P, -1, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = cv2.boxFilter(a, -1, r) / N
    meanB = cv2.boxFilter(b, -1, r) / N

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p = p[:,:,np.newaxis]

    out = np.zeros_like(p)
    for ch in range(p.shape[2]):
        out[:,:,ch] = _gf_colorgray(I, p[:,:,ch], r, eps)
    return np.squeeze(out) if p.ndim == 2 else out

