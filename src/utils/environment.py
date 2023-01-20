import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def __hostname():
    # env variables set up wrong on aimscdt2.dns.eng.ox.ac.uk
    # if 'HOST' in os.environ:
    #     return str(os.environ['HOST'])
    # if 'HOSTNAME' in os.environ:
    #     return str(os.environ['HOSTNAME'])
    # else:
    return str(subprocess.check_output('hostname', shell=True).decode().strip())


@lru_cache(None)
def user():
    if 'USER' in os.environ:
        return str(os.environ['USER'])
    else:
        return str(subprocess.check_output('whoami', shell=True).decode().strip())


def is_slurm():
    return 'SLURM_JOB_ID' in os.environ and os.environ['SLURM_JOB_NAME'] not in  ['zsh', 'bash']


def get_slurm_id():
    return os.environ.get('SLURM_JOB_ID', None)


def is_aims_machine():
    hostname = __hostname()
    return 'aims' in hostname


def is_vggdev_machine():
    hostname = __hostname()
    return 'vggdev' in hostname or 'vggdebug' in hostname


def can_fit_in_tmp(path):
    tmp_avail = int(str(subprocess.check_output(['/usr/bin/df', '-k', '--output=avail', str(os.environ['TMPDIR'])],
                                                close_fds=True).decode().strip()).split()[-1].strip()) * 1024
    path_size = int(Path(path).stat().st_size)
    print(f"{Path(path).name} size {path_size / 2 ** 30:.2f}GB vs {tmp_avail / 2 ** 30:.2f}GB")
    return path_size < tmp_avail


def check_user(username, partial=True):
    username = username.lower()
    run_user = user().lower()
    if partial:
        return username in run_user
    return username == run_user
