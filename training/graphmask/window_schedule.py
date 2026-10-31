"""Window order is independent of model, sampler and gate random streams."""
import random
import numpy as np

def window_order(count, order_seed, epoch, step):
    order=list(range(count))
    if order_seed is not None:
        random.Random(f'window-order:{order_seed}:{epoch}:{step}').shuffle(order)
    return order

def batch_seed(model_seed, epoch, step, window):
    return int(np.random.SeedSequence([model_seed,epoch,step,window]).generate_state(1)[0])
