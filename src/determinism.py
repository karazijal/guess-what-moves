import os
lvl = int(os.environ.get('TRY_DETERMISM_LVL', '0'))
if lvl > 0:
    print(f'Attempting to enable deterministic cuDNN and cuBLAS operations to lvl {lvl}')
if lvl >= 2:
    # turn on deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  #Need to set before torch gets loaded
    import torch
    # Since using unstable torch version, it looks like 1.12.0.devXXXXXXX
    if torch.version.__version__ >= '1.12.0':
        torch.use_deterministic_algorithms(True, warn_only=(lvl < 3))
    elif lvl >= 3:
        torch.use_deterministic_algorithms(True)  # This will throw errors if implementations are missing
    else:
        print(f"Torch verions is only {torch.version.__version__}, which will cause errors on lvl {lvl}")
if lvl >= 1:
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def i_do_nothing_but_dont_remove_me_otherwise_things_break():
    """This exists to prevent formatters from treating this file as dead code"""
    pass
