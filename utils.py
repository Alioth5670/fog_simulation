import cv2
import numpy as np

def clip_to_unit_range(image):
    return np.clip(image, 0, 1)

def darkChannel(img,r = 15):
    '''
    @Description :求取暗通道图
                  np.min实现三通道中的最小值;cv2.erode实现了窗口中的最小值 
    @Parameter   :img为原始图像
    @Parameter   :r  为窗口大小
    @Return      :原始图像的暗通道图
    '''
    DarkChann = cv2.erode(np.min(img,2),np.ones((r,r)))

    return DarkChann

def estimateA(img,darkChann):
    '''
    @Description :将按通道中前0.1%亮度的像素定位到原图中,并在这些位置上求取各个通道的均值以作为三通道的全局大气光A   
    @Parameter   :img为原始图像
    @Parameter   :darkChann为暗通道图
    @Return      :三通道下的A
    '''

    h,w,_  = img.shape
    length = h*w
    num    = max(int(length *0.0001),1)
    DarkChannVec = np.reshape(darkChann,length)  # convert to a row vector
    index  = DarkChannVec.argsort()[length-num:]
    rowIdx = index // w
    colIdx = index %  w
    coords = np.stack((colIdx,rowIdx),axis = 1)

    sumA   = np.zeros((1,1,3))
    for coord in coords:
        col,row = coord
        sumA    += img[row,col,:]
    A = sumA / num

    return A 

def haze_linear(R, t, L):
    """
    Generate hazy image from clean image using the linear haze model corresponding 
    to Lambert-Beer law.
    
    Args:
    - R: H-by-W-by-C clean image array representing true radiance of the scene.
    - t: H-by-W transmission map array.
    - L: 1-by-1-by-C array representing the homogeneous atmospheric light.
    
    Returns:
    - I: Synthetic hazy image with the same size as the input clean image R.
    """
    
    # 获取图像通道数
    image_channels = L.shape[2]

    # 复制传输映射至所有通道
    t_replicated = np.repeat(t[...], image_channels, axis=-1)
    # 应用线性雾霾模型
    I = t_replicated * R + (1 - t_replicated) * np.tile(L, (t.shape[0], t.shape[1], 1))

    return I