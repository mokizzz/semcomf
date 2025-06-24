import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(
        "Warning: Matplotlib not found. Some visualization functions may not be available."
    )


from .image_utils import convert_pil_to_tensor, convert_tensor_to_pil, resize_pil_image


def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    """Converts various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, torch.Tensor):
        return convert_tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported image type for visualization: {type(image)}")


def display_images(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    titles: Optional[List[str]] = None,
    max_cols: int = 5,
    figsize_per_image: Tuple[int, int] = (4, 4),
    save_path: Optional[str] = None,
):
    """
    Displays a list of images in a grid using Matplotlib.
    Images can be PIL.Image, PyTorch Tensor, or NumPy array.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is required for display_images function. Skipping.")
        if save_path and len(images) == 1:
            try:
                pil_img = _to_pil(images[0])
                pil_img.save(save_path)
                print(f"Saved image to {save_path} as Matplotlib is unavailable.")
            except Exception as e:
                print(f"Could not save image: {e}")
        return

    num_images = len(images)
    if num_images == 0:
        return

    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1])
    )
    axes = np.array(axes).flatten()

    for i, img_data in enumerate(images):
        try:
            pil_img = _to_pil(img_data)
            axes[i].imshow(pil_img)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis("off")
        except Exception as e:
            print(f"Error displaying image {i}: {e}")
            axes[i].text(0.5, 0.5, "Error", ha="center", va="center")
            axes[i].axis("off")

    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def arrange_images_grid(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    target_image_size: Optional[Tuple[int, int]] = None,
    grid_fill_color: Union[str, Tuple[int, int, int]] = "black",
    resize_interpolation=Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Arranges multiple images into a single grid image.
    Images are placed as close to a square grid as possible.
    None values or unprocessable images are replaced with `grid_fill_color`.
    """
    if not images:
        return Image.new(
            "RGB",
            target_image_size if target_image_size else (256, 256),
            grid_fill_color,
        )

    processed_pil_images = []
    first_valid_image = None

    for img_data in images:
        if img_data is None:
            processed_pil_images.append(None)
            continue
        try:
            pil_img = _to_pil(img_data)
            if target_image_size:
                pil_img = resize_pil_image(
                    pil_img, target_image_size, interpolation=resize_interpolation
                )
            processed_pil_images.append(pil_img)
            if first_valid_image is None:
                first_valid_image = pil_img
        except Exception as e:
            print(
                f"Warning: Could not process an image for grid arrangement: {e}. Using fill color."
            )
            processed_pil_images.append(None)

    if first_valid_image is None:
        w, h = target_image_size if target_image_size else (256, 256)
        return Image.new(
            "RGB",
            (
                w * math.ceil(math.sqrt(len(images))),
                h * math.ceil(math.sqrt(len(images))),
            ),
            grid_fill_color,
        )

    img_w, img_h = first_valid_image.size

    for i in range(len(processed_pil_images)):
        if processed_pil_images[i] is None:
            processed_pil_images[i] = Image.new("RGB", (img_w, img_h), grid_fill_color)

    num_images = len(processed_pil_images)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)

    combined_width = grid_cols * img_w
    combined_height = grid_rows * img_h
    combined_image = Image.new(
        "RGB", (combined_width, combined_height), grid_fill_color
    )

    for i, img in enumerate(processed_pil_images):
        row_idx = i // grid_cols
        col_idx = i % grid_cols
        combined_image.paste(img, (col_idx * img_w, row_idx * img_h))

    return combined_image


def arrange_images_side_by_side(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    target_image_height: Optional[int] = None,
    separator_width: int = 10,
    separator_color: Union[str, Tuple[int, int, int, int]] = (
        0,
        0,
        0,
        0,
    ),
    background_color: Union[str, Tuple[int, int, int, int]] = (
        255,
        255,
        255,
        0,
    ),
    resize_interpolation=Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Arranges images side-by-side horizontally with a separator.
    If target_image_height is provided, all images are resized to this height, maintaining aspect ratio.
    """
    if not images:
        return Image.new(
            "RGBA",
            (100, target_image_height if target_image_height else 100),
            background_color,
        )

    processed_pil_images = []
    max_h = 0

    for img_data in images:
        if img_data is None:
            if target_image_height:
                placeholder = Image.new(
                    "RGBA", (target_image_height, target_image_height), separator_color
                )
                processed_pil_images.append(placeholder)
                max_h = max(max_h, target_image_height)
            continue

        try:
            pil_img = _to_pil(img_data).convert("RGBA")
            if target_image_height:
                orig_w, orig_h = pil_img.size
                new_w = int(orig_w * (target_image_height / orig_h))
                pil_img = resize_pil_image(
                    pil_img,
                    (new_w, target_image_height),
                    interpolation=resize_interpolation,
                )
            processed_pil_images.append(pil_img)
            max_h = max(max_h, pil_img.height)
        except Exception as e:
            print(
                f"Warning: Could not process an image for side-by-side arrangement: {e}"
            )

    if not processed_pil_images:
        return Image.new(
            "RGBA",
            (100, target_image_height if target_image_height else 100),
            background_color,
        )

    total_width = sum(img.width for img in processed_pil_images) + separator_width * (
        len(processed_pil_images) - 1
    )

    if total_width <= 0:
        total_width = 100
    canvas_height = target_image_height if target_image_height else max_h
    if canvas_height <= 0:
        canvas_height = 100

    combined_image = Image.new("RGBA", (total_width, canvas_height), background_color)
    draw = ImageDraw.Draw(combined_image)

    current_x = 0
    for i, img in enumerate(processed_pil_images):
        y_offset = (canvas_height - img.height) // 2
        combined_image.paste(img, (current_x, y_offset), img)
        current_x += img.width
        if i < len(processed_pil_images) - 1:
            separator_rect = (current_x, 0, current_x + separator_width, canvas_height)
            draw.rectangle(separator_rect, fill=separator_color)
            current_x += separator_width

    return combined_image


if __name__ == "__main__":
    # Create some dummy images for testing
    img_list = [
        Image.new("RGB", (100, 150), "red"),
        torch.rand(3, 120, 80),
        np.random.randint(0, 255, size=(180, 120, 3), dtype=np.uint8),
        Image.new("RGB", (90, 90), "blue"),
        None,
        Image.new("L", (50, 50), "white"),
    ]

    # Test display_images
    if MATPLOTLIB_AVAILABLE:
        print("Testing display_images...")
        display_images(
            img_list,
            titles=["Red", "RandTensor", "RandNumpy", "Blue", "None", "Gray"],
            max_cols=3,
        )
        # display_images([img_list[0]], save_path="single_display_test.png")
    else:
        print("Skipping display_images test as Matplotlib is not available.")

    # Test arrange_images_grid
    print("\nTesting arrange_images_grid...")
    grid_img = arrange_images_grid(img_list, target_image_size=(100, 100))
    if grid_img:
        grid_img.save("test_grid_arrangement.png")

    grid_img_no_resize = arrange_images_grid(
        [Image.new("RGB", (50, 50), "cyan"), Image.new("RGB", (50, 50), "magenta")]
    )
    if grid_img_no_resize:
        grid_img_no_resize.save("test_grid_no_resize.png")

    # Test arrange_images_side_by_side
    print("\nTesting arrange_images_side_by_side...")
    side_by_side_img = arrange_images_side_by_side(img_list, target_image_height=100)
    if side_by_side_img:
        side_by_side_img.save("test_side_by_side_arrangement.png")

    side_by_side_img_no_resize = arrange_images_side_by_side(
        [Image.new("RGB", (50, 100), "orange"), Image.new("RGB", (80, 60), "purple")]
    )
    if side_by_side_img_no_resize:
        side_by_side_img_no_resize.save("test_side_by_side_no_resize.png")

    print("\nTest complete. Check for output images.")
