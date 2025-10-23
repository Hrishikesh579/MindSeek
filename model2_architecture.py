import torch
import torch.nn as nn
from transformers import BertModel

class MentalHealthClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(MentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_text_model(num_labels=4):
    from .model2_architecture import MentalHealthClassifier
    return MentalHealthClassifier(num_labels=num_labels)