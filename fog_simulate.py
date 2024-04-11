
import numpy as np

from utils import darkChannel, estimateA,haze_linear, clip_to_unit_range
from guid_filter import guided_filter



def simulate(depth_map,image_init, beta=0.01, window_size_guidedFilter=41, window_size_darkChannel=15,epsilon=1e-6):
    '''
    depth_map: absolute depth map for image
    image_init: init image without fog
    '''
    # 透射率计算
    t_initial = np.exp(-beta * depth_map)

    t = clip_to_unit_range(guided_filter(image_init,t_initial, r=window_size_guidedFilter, eps=epsilon))

    #get dark cheannel
    dark_channel = darkChannel(image_init, window_size_darkChannel)

    #light estimate
    L_atm = estimateA(image_init,dark_channel)

    haze_image = haze_linear(image_init, t, L_atm)

    return haze_image






