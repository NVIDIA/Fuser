import sys
import torch
from functools import partial

from transformers.models.phi3 import Phi3PreTrainedModel

from .model_configs import configs

def hf_qwen2(dtype):

    class MyModel(Phi3PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    
            # Initialize weights and apply final processing
            self.post_init()
    
        def forward(self, input_ids: torch.LongTensor):
            inputs_embeds = self.embed_tokens(input_ids)
            return (inputs_embeds,)

    config = configs["hf_qwen2"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda', requires_grad=False)
        return {"input_ids": input_ids}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda', dtype=dtype, requires_grad=False)
        return grad

    def iobytes():
        # TODO: compute iobytes
        return 1

    return MyModel(config).cuda(), partial(inputs, dtype), partial(grads, dtype), iobytes

embedding_setup = {
    "hf_qwen2": hf_qwen2,
    #"hf_phi3": hf_phi3,
    #"hf_mistral_nemo": hf_mistral_nemo,
}
