"""Where training runs. Add one by subclassing training.base.Backend and listing it here.

Kaggle, SSH, Modal and This Mac train voices; the pay-as-you-go GPU services (machines.PaidCloud: Runpod,
Vast.ai, Thunder Compute, Lambda, Lightning AI, Beam, Cerebrium, Hugging Face Jobs) are for Create in the cloud.
"""
from .cerebrium_cloud import Cerebrium
from .hf_jobs import HFJobs
from .kaggle import Kaggle
from .lambda_cloud import Lambda
from .local_mac import LocalMac
from .modal_cloud import Modal
from .runpod import Runpod
from .sdk_cloud import Beam, Lightning
from .ssh import SSH
from .thunder import Thunder
from .vast import Vast

BACKENDS = {b.id: b for b in (Kaggle(), SSH(), Modal(), Runpod(), Vast(), Thunder(), Lambda(), Lightning(), Beam(),
                              Cerebrium(), HFJobs(), LocalMac())}
# the services Create in the cloud can use, in the order the menu lists them
CLOUDS = ("kaggle", "modal", "runpod", "vast", "thunder", "lambda", "lightning", "beam", "cerebrium", "hfjobs")
