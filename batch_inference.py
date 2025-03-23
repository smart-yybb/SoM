import os
import torch
from PIL import Image
from segment_anything import sam_model_registry
# sam
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto

sam_ckpt = "./sam_vit_h_4b8939.pth"
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()

input_path = './inputs/'
output_path = './outputs/'

os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)


# 获取 path 下的所有文件的绝对地址
def recursive_get_image_paths(path):
    if not os.path.exists(path):
        raise Exception(f"图片路径有错，路径:{path}")
    images = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            temp = recursive_get_image_paths(file_path)
            images = images + temp
        elif file_path.find('.png') != -1 or file_path.find('.jpg') != -1:
            images.append(file_path)
    return images


@torch.no_grad()
def inference(image, alpha, label_mode, anno_mode):
    global history_images;
    history_images = []
    global history_masks;
    history_masks = []

    _image = image.convert('RGB')
    # _mask = image['layers'][0].convert('L') if image['layers'] else None

    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale = 640, 100, 100
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output, mask = inference_sam_m2m_auto(model_sam, _image, text_size, label_mode, alpha, anno_mode)
        # convert output to PIL image
        history_masks.append(mask)
        history_images.append(Image.fromarray(output))
        return (output, [])


if __name__ == "__main__":
    image_paths = recursive_get_image_paths(input_path)
    alpha = 0.05
    label_mode = "1"
    anno = "Mark"
    print(f"Start to infer {len(image_paths)} images...\n")
    count = 0
    for image_path in image_paths:
        count += 1
        print(f"Handling the {count}th image\n")
        img = Image.open(image_path)
        (output, _) = inference(img, alpha, label_mode, anno)
        im_pil = Image.fromarray(output)  # 转换为 PIL 格式
        im_pil.save(os.path.join(output_path, os.path.basename(image_path)))  # 保存为 PNG 或 JPG
    print("Completed")
