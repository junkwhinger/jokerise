from celluloid import Camera
import matplotlib.pyplot as plt


def save_video(frame_list, filename, fps):
    fig, ax = plt.subplots()
    camera = Camera(fig)

    # h, w, _ = frame_list[0].shape
    # fig.set_size_inches(w,h)

    for frame in frame_list:

        # BGR to RGB
        ax.imshow(frame[:, :, ::-1])
        # ax.axis("off")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        plt.tight_layout()
        camera.snap()

        # plt.savefig("dd.png")
        # raise ValueError

    interval_value = 1000 / fps
    animation = camera.animate(interval=interval_value, blit=True)
    animation.save(filename,
                   dpi=100,
                   savefig_kwargs={
                       'facecolor': 'none',
                       'pad_inches': 'tight'
                   })
