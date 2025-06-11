from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    """Abstract base for all architectures with shared heads."""
    def __init__(self, hidden_size, relation_vocab_size, entity_vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_vocab_size = relation_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.relation_head = nn.Linear(hidden_size, relation_vocab_size)
        self.entity_head   = nn.Linear(hidden_size, entity_vocab_size)

    @abstractmethod
    def forward(self, input_ids, attention_mask, labels=None):
        """Returns (rel_logits, ent_logits) and optionally loss."""
        pass