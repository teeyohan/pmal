from architecture.abstract_arch import AbsArchitecture
from architecture.GCN import GCN
from architecture.GAT import GAT
from architecture.GIN import GIN
from architecture.Cross_stitch import Cross_stitch
from architecture.MTAN import MTAN
from architecture.MMoE import MMoE
from architecture.LTB import LTB
from architecture.DSelect_k import DSelect_k
from architecture.ETR_NLP import ETR_NLP
from architecture.Graphormer import Graphormer


__all__ = ['AbsArchitecture',
           'GCN',
           'GAT',
           'GIN',
           'Cross_stitch',
           'MTAN',
           'MMoE',
           'LTB',
           'DSelect_k',
           'ETR_NLP',
           'Graphormer']