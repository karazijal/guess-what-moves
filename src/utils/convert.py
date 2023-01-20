import torch
import itertools

def cast_like(maybe_tensor, example_tensor):
    if not torch.is_tensor(maybe_tensor):
        maybe_tensor = torch.tensor(maybe_tensor)
    maybe_tensor = maybe_tensor.to(example_tensor.device).to(example_tensor.dtype)
    shape = [*maybe_tensor.shape, *[1] * len(example_tensor.shape)]
    if not shape:
        shape = [1]
    return maybe_tensor.view(*shape)


def lofd_2_dofl(list_of_dicts, make_tensor=True):
    keys = set(itertools.chain.from_iterable(list_of_dicts))
    out_dict = {}
    for k in keys:
        out_dict[k] = [d[k] for d in list_of_dicts if k in d]
        if make_tensor:
            example_tensor = next((v for v in out_dict[k] if torch.is_tensor(v)), None)
            if example_tensor is None:
                out_dict[k] = torch.tensor(out_dict[k])
            else:
                out_dict[k] = torch.cat([cast_like(t, example_tensor) for t in out_dict[k]], 0)
    return out_dict
