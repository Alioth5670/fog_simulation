import cv2
from fog_simulate import *
import time



#########################################################
beta = 0.05  #调整雾的浓度，不要小于0.003
#########################################################
epsilon = 1e-6
window_size_guidedFilter = 41
window_size_darkChannel = 15

# (1080*1920) time: 1.556933879852295
START = time.time()
image_name = "00000000"
image_file_path = f'./input/{image_name}.png'
depth_map_file_path = f'./input/{image_name}_depth.png'

init_image = cv2.imread(image_file_path, cv2.COLOR_BGR2RGB)/255.0
depth_map_uint16 = cv2.imread(depth_map_file_path, cv2.IMREAD_UNCHANGED)
depth_map_float = depth_map_uint16.astype(np.float32) / 65535.0 

# 使用视差图估计绝对深度图
# 使用Depth_anything估计得到的视差图 @https://github.com/LiheYoung/Depth-Anything
L = 0.07 / np.clip(depth_map_float,  epsilon, 1) 

haze_image = simulate(L,init_image,\
                      beta=beta,\
                      window_size_guidedFilter=window_size_guidedFilter,\
                      window_size_darkChannel=window_size_darkChannel)

cv2.imwrite(f'./output/result_{beta}.png', haze_image*255)
print(time.time()-START)
