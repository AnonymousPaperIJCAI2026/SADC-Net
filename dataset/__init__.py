"""Dataset loaders for source adaptation and cross-domain evaluation."""

from .brainMRI import BrainMRIDataset
from .br35h import Br35HDataset
from .btad import BTADDataset
from .clinicdb import ClinicDBDataset
from .colondb import ColonDBDataset
from .dagm import DAGMDataset
from .dtd import DTDDataset
from .endo import EndoDataset
from .isic import ISICDataset
from .kvasir import KvasirDataset
from .mpdd import MpDDDataset
from .mvtec import MVTecDataset
from .mvtec3d_few_shot import MVTec3DDataset
from .tn3k import TN3KDataset
from .visa import VisaDataset


DATASETS = {
    "mvtec": MVTecDataset,
    "visa": VisaDataset,
    "btad": BTADDataset,
    "mpdd": MpDDDataset,
    "dtd": DTDDataset,
    "dagm": DAGMDataset,
    "mvtec3d": MVTec3DDataset,
    "brainmri": BrainMRIDataset,
    "br35h": Br35HDataset,
    "isic": ISICDataset,
    "clinic": ClinicDBDataset,
    "colon": ColonDBDataset,
    "kvasir": KvasirDataset,
    "endo": EndoDataset,
    "tn3k": TN3KDataset,
}

IMAGE_ONLY = {"brainmri", "br35h"}
PIXEL_ONLY = {"isic", "clinic", "colon", "kvasir", "endo", "tn3k"}


def load_dataset(name, root, image_transform, mask_transform):
    """Load the labeled benchmark partition used in the transfer protocol."""
    return DATASETS[name](
        root=root, train=False, category=None,
        transform=image_transform, gt_target_transform=mask_transform,
    )
