# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
import gradio as gr
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np

from gpt4v import request_gpt4v
from openai import OpenAI

import matplotlib.colors as mcolors

css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

import os
from datetime import datetime
import logging

# init log
start_timestamp = datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S')
log_path = f"./{start_timestamp}.log"
log_format = '[%(levelname)s] %(asctime)s - %(message)s'
log_level = logging.INFO
logging.basicConfig(level=log_level, format=log_format, filename=log_path, filemode='w')

client = OpenAI()

'''
build args
'''
sam_ckpt = "./sam_vit_h_4b8939.pth"

'''
build model
'''
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()

history_images = []
history_masks = []
history_texts = []


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    # global history_images;
    # history_images = []
    # global history_masks;
    # history_masks = []

    _image = image['background'].convert('RGB')
    _mask = image['layers'][0].convert('L') if image['layers'] else None

    model_name = 'sam'

    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale = 640, 100, 100
    text, text_part, text_thresh = '', '', '0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic = False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(_mask))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i + 1) for i in range(num_features)])

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha,
                                                             anno_mode)

        # convert output to PIL image
        history_masks.append(mask)
        history_images.append(Image.fromarray(output))
        return (output, [])


def gpt4v_response(message, history):
    global history_images
    global history_texts;
    history_texts = []
    try:
        if len(history_images) == 0:
            return "请先上传图片并且运行"
        logging.info(f"gpt4v_response msg:{message}\n")
        res = request_gpt4v(message, history_images[0])
        logging.info(f"gpt4v_response res:{res}\n")
        history_texts.append(res)
        return res
    except Exception as e:
        logging.error(e)
        return None


def highlight(mode, alpha, label_mode, anno_mode, *args, **kwargs):
    res = history_texts[0]
    # find the seperate numbers in sentence res
    res = res.split(' ')
    res = [r.replace('.', '').replace(',', '').replace(')', '').replace('"', '') for r in res]
    # find all numbers in '[]'
    res = [r for r in res if '[' in r]
    res = [r.split('[')[1] for r in res]
    res = [r.split(']')[0] for r in res]
    res = [r for r in res if r.isdigit()]
    res = list(set(res))
    sections = []
    for i, r in enumerate(res):
        mask_i = history_masks[0][int(r) - 1]['segmentation']
        sections.append((mask_i, r))
    return (history_images[0], sections)


'''
launch app
'''

demo = gr.Blocks()
image = gr.ImageMask(label="Input", type="pil", sources=["upload"], interactive=True,
                     brush=gr.Brush(colors=["#FFFFFF"]))
slider = gr.Slider(1, 3, value=1.8,
                   label="Granularity")  # info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]"
mode = gr.Radio(['Automatic', 'Interactive', ], value='Automatic', label="Segmentation Mode")
anno_mode = gr.CheckboxGroup(choices=["Mark", "Mask", "Box"], value=['Mark'], label="Annotation Mode")
image_out = gr.AnnotatedImage(label="SoM Visual Prompt", height=512)
runBtn = gr.Button("Run")
highlightBtn = gr.Button("Highlight")
bot = gr.Chatbot(label="GPT-4V + SoM", height=256)
slider_alpha = gr.Slider(0, 1, value=0.05, label="Mask Alpha")  # info="Choose in [0, 1]"
label_mode = gr.Radio(['Number', 'Alphabet'], value='Number', label="Mark Mode")

title = "Set-of-Mark (SoM) Visual Prompting for Extraordinary Visual Grounding in GPT-4V"
description = "This is a demo for SoM Prompting to unleash extraordinary visual grounding in GPT-4V. Please upload an image and them click the 'Run' button to get the image with marks. Then chat with GPT-4V below!"

with demo:
    gr.Markdown(
        "<h1 style='text-align: center'><img src='https://som-gpt4v.github.io/website/img/som_logo.png' style='height:50px;display:inline-block'/>  Set-of-Mark (SoM) Prompting Unleashes Extraordinary Visual Grounding in GPT-4V</h1>")
    # gr.Markdown("<h2 style='text-align: center; margin-bottom: 1rem'>Project: <a href='https://som-gpt4v.github.io/'>link</a>     arXiv: <a href='https://arxiv.org/abs/2310.11441'>link</a>     Code: <a href='https://github.com/microsoft/SoM'>link</a></h2>")
    with gr.Row():
        with gr.Column():
            image.render()
            slider.render()
            with gr.Accordion("Detailed prompt settings (e.g., mark type)", open=False):
                with gr.Row():
                    mode.render()
                    anno_mode.render()
                with gr.Row():
                    slider_alpha.render()
                    label_mode.render()
        with gr.Column():
            image_out.render()
            runBtn.render()
            highlightBtn.render()
    with gr.Row():
        gr.ChatInterface(chatbot=bot, fn=gpt4v_response)

    runBtn.click(inference, inputs=[image, slider, mode, slider_alpha, label_mode, anno_mode],
                 outputs=image_out)
    highlightBtn.click(highlight, inputs=[image, mode, slider_alpha, label_mode, anno_mode],
                       outputs=image_out)

demo.queue().launch(share=True, server_port=6093)
