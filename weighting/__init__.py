from weighting.abstract_weighting import AbsWeighting
from weighting.EW import EW
from weighting.UW import UW
from weighting.MGDA import MGDA
from weighting.GradNorm import GradNorm
from weighting.DWA import DWA
from weighting.PCGrad import PCGrad
from weighting.CAGrad import CAGrad
from weighting.GradVac import GradVac
from weighting.Align import Align
from weighting.Excess import Excess


__all__ = ['AbsWeighting',
           'EW',
           'UW',
           'MGDA',
           'GradNorm',
           'DWA',
           'PCGrad',
           'CAGrad',
           'GradVac',
           'Align',
           'Excess']