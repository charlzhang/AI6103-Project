from PIL import Image
import numpy as np
from utils.common import tensor2im, im2tensor
from torchvision import transforms
import torch
import torch.nn.functional as F

def display_alongside_image(source_image, target_image,result_image):
    resize_dims = (256,256)
    res = np.concatenate([np.array(source_image),
                          np.array(target_image),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def face_to_latents(images,net):
    codes =  net.encoder(images.to("cuda").float())
    if codes.ndim == 2:
        codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
    else:
        codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes

def latents_to_face(latents,net):
    images,_ = net.decoder([latents.float()],input_is_latent=True,randomize_noise=False,return_latents=False)
    return images

# 定义转换操作，包括缩放和转换为张量
resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放图像
    transforms.ToTensor()           # 转换回张量
])

def resize_images(imgs):
    # 初始化一个空的张量列表来存储处理后的图像
    processed_imgs = []

    # 遍历每张图像
    for img in imgs:
        # 将张量转换为PIL图像
        pil_img = tensor2im(img)
        
        # 应用缩放并转换回张量
        resized_img_tensor = resize_transform(pil_img)
        
        # 添加到列表中
        processed_imgs.append(resized_img_tensor)
    
    # 将列表中的张量堆叠成一个新的批次张量
    return torch.stack(processed_imgs).to('cuda')

def resize_tensor(input_tensor, height, width):
    # 使用双线性插值进行大小调整
    resized_tensor = F.interpolate(input_tensor, size=(height, width), mode='bilinear', align_corners=False)
    return resized_tensor