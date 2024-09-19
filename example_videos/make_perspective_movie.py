import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

src_points = (
    np.array(
        [
            [50, 50],
            [500, 400],
            [800, 1000],
            [100, 600],
        ]
    )
    / 1000
)
src_points = src_points.astype(np.float32)
dst_points = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ]
).astype(np.float32)

test_points = np.random.uniform(np.array([0.2, 0.2]), np.array([0.7, 0.7]), (10, 2))


def transform(src, dst, pts) -> np.ndarray:
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    H = cv2.getPerspectiveTransform(src, dst)
    mul = np.einsum("ij,bj->bi", H, pts)
    return mul[:, :2] / mul[:, 2:3]


def get_frame_buffer(figure: mpl.figure.Figure) -> np.ndarray:
    """Returns a BGR image for opencv from a matplotlib figure"""
    figure.canvas.draw()
    buf = figure.canvas.buffer_rgba()
    width, height = figure.canvas.get_width_height()
    return np.frombuffer(buf, np.uint8).reshape(height * 2, width * 2, 4)[..., 2::-1]


writer = None

for t in tqdm(np.linspace(0, 1, 60, endpoint=True)):
    mod_dst_pts = (1 - t) * src_points + t * dst_points
    transformed = transform(src_points, mod_dst_pts, test_points)

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(4):
        ax.plot(
            [mod_dst_pts[i, 0], mod_dst_pts[(i + 1) % 4, 0]],
            [mod_dst_pts[i, 1], mod_dst_pts[(i + 1) % 4, 1]],
            color="red",
            linewidth=2,
        )

    ax.scatter(transformed[:, 0], transformed[:, 1], cmap="tab10", c=np.arange(10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    frame = get_frame_buffer(fig)

    if writer is None:
        writer = cv2.VideoWriter(
            "perspective_video.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,  # skipping 1/5 frames
            (frame.shape[1], frame.shape[0]),
            isColor=True,
        )

    writer.write(frame)


writer.release()
