import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate_white_image(output_path: str | Path = "white_image.png") -> None:
    """Generate and save a 640x480 pure white image."""
    image = Image.new("RGB", (640, 480), "white")
    image.save(output_path)


def draw_square(
    image: Image.Image,
    x: int,
    y: int,
    color: str,
    width: int = 50,
    height: int = 50,
) -> Image.Image:
    """Place a rectangle with the specified color on the image."""
    ImageDraw.Draw(image).rectangle(
        (x, y, x + width - 1, y + height - 1), fill=color
    )
    return image


def draw_red_square(
    image: Image.Image, x: int, y: int, width: int = 50, height: int = 50
) -> Image.Image:
    """Place a red rectangle on the image."""
    return draw_square(image, x, y, "red", width, height)


def draw_blue_square(
    image: Image.Image, x: int, y: int, width: int = 50, height: int = 50
) -> Image.Image:
    """Place a blue rectangle on the image."""
    return draw_square(image, x, y, "blue", width, height)


def draw_green_square(
    image: Image.Image, x: int, y: int, width: int = 50, height: int = 50
) -> Image.Image:
    """Place a green rectangle on the image."""
    return draw_square(image, x, y, "green", width, height)


def draw_obstacle(
    image: Image.Image, x: int, y: int, width: int = 50, height: int = 50
) -> Image.Image:
    """Place a black rectangular obstacle on the image."""
    return draw_square(image, x, y, "black", width, height)


def generate_scene(
    output_path: str | Path | None = None,
    seed: int | None = None,
    box_width: int = 50,
    box_height: int = 50,
    obstacle_width: int = 50,
    obstacle_height: int = 50,
) -> None:
    """Generate and save a scene with colored boxes and black obstacles."""
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)

    rng = random.Random(seed)
    image = Image.new("RGB", (640, 480), "white")
    objects = []
    obstacles = []
    occupied_bboxes = []
    box_size = (box_width, box_height)
    obstacle_size = (obstacle_width, obstacle_height)

    def random_position(width: int, height: int) -> tuple[int, int]:
        return (
            rng.randint(0, image.width - width),
            rng.randint(0, image.height - height),
        )

    def overlaps_existing(x: int, y: int, width: int, height: int) -> bool:
        """Return whether a rectangle overlaps any previously placed rectangle."""
        for existing_x, existing_y, existing_width, existing_height in occupied_bboxes:
            if (
                x < existing_x + existing_width
                and x + width > existing_x
                and y < existing_y + existing_height
                and y + height > existing_y
            ):
                return True
        return False

    box_drawers = [draw_red_square, draw_blue_square, draw_green_square]

    for object_id in range(rng.randint(2, 5)):
        for _ in range(5):
            x, y = random_position(*box_size)
            if not overlaps_existing(x, y, *box_size):
                break
        else:
            continue

        color, draw_box = rng.choice(
            [("red", draw_red_square), ("blue", draw_blue_square), ("green", draw_green_square)]
        )
        draw_box(image, x, y, *box_size)
        occupied_bboxes.append((x, y, *box_size))
        objects.append(
            {
                "id": f"obj_{object_id}",
                "type": "box",
                "property": {"color": color},
                "bbox": {"x": x, "y": y, "width": box_width, "height": box_height},
            }
        )

    for obstacle_id in range(rng.randint(1, 3)):
        for _ in range(5):
            x, y = random_position(*obstacle_size)
            if not overlaps_existing(x, y, *obstacle_size):
                break
        else:
            continue

        draw_obstacle(image, x, y, *obstacle_size)
        occupied_bboxes.append((x, y, *obstacle_size))
        obstacles.append(
            {
                "id": f"obs_{obstacle_id}",
                "bbox": {"x": x, "y": y, "width": obstacle_width, "height": obstacle_height},
            }
        )

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "output" / f"scene_{seed}.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    ground_truth = {
        "image": output_path.name,
        "objects": objects,
        "obstacles": obstacles,
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    print("{} is generated".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    generate_scene(args.output, args.seed)
