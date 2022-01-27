from fairseq.models import register_model_architecture
from .adv_model.model import AdvbertModel
from .adv_masked_lm import AdvMaskedLmLoss
from .adv_masked_lm_task import AdvMaskedLMTask
