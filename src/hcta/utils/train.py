import numpy as np


def print_status_bar(iteration, total, loss, metrics=None):
    """Print the status bar during training"""
    metrics = " - ".join(
        ["{}: {:e}".format(m.name, m.result()) for m in loss + (metrics or [])]
    )
    # end = "" if iteration < total else "\n"
    end = "\n"
    print("Epoch \r{}/{} - ".format(iteration, total) + metrics, end=end)


def sample(x_domain, t_domain, batch_size):
    x = np.random.uniform(low=x_domain[0], high=x_domain[1], size=(batch_size, 1))
    t = np.random.uniform(low=t_domain[0], high=t_domain[1], size=(batch_size, 1))
    sampled_input = np.hstack((x, t))
    return sampled_input
