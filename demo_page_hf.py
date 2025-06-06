"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

from utils.utils import *


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)

        # Set device and precision based on availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id_or_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        self.tokenizer = self.processor.tokenizer

    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)"""
        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.to(self.device)

        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(prompts, add_special_tokens=False, return_tensors="pt")

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)

        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
        )

        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)

        return results[0] if not is_batch else results


def process_page(image_path, model, save_dir, max_batch_size=None):
    """Parse document images with two stages"""
    pil_image = Image.open(image_path).convert("RGB")
    layout_output = model.chat("Parse the reading order of this document.", pil_image)

    padded_image, dims = prepare_image(pil_image)
    recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size)

    json_path = save_outputs(recognition_results, image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size=None):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0

    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    figure_results.append({
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "text": "",
                        "reading_order": reading_order,
                    })
                else:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    if label == "tab":
                        table_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()

    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)

    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    results = []
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)

    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i + batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)
        batch_results = model.chat(prompts_list, crops_list)

        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Document processing tool using DOLPHIN model")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image or directory of images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum number of document elements to parse in a single batch (default: 16)",
    )
    args = parser.parse_args()

    model = DOLPHIN(args.model_path)

    if os.path.isdir(args.input_path):
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            image_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        image_files = sorted(image_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        image_files = [args.input_path]

    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    setup_output_dirs(save_dir)

    total_samples = len(image_files)
    print(f"\nTotal samples to process: {total_samples}")

    for image_path in image_files:
        print(f"\nProcessing {image_path}")
        try:
            json_path, recognition_results = process_page(
                image_path=image_path,
                model=model,
                save_dir=save_dir,
                max_batch_size=args.max_batch_size,
            )
            print(f"Processing completed. Results saved to {save_dir}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
