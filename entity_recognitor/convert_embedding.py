import flair
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import inspect
import torch


model_path = "./checkpoints/best-model.pt"
model = SequenceTagger.load(model_path)
embedding = model.embeddings
assert isinstance(embedding, (TransformerWordEmbeddings))

sentences = [Sentence("to speak to a customer service advisor"), Sentence("to speak to a customer")]


# model.embeddings = model.embeddings.export_onnx("flert-embeddings_2.onnx", sentences, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

example_tensors = embedding.prepare_tensors(sentences)
dynamic_axes = {"input_ids": {0: 'batch', 1: "seq_length"},
                "token_lengths": {0: 'sent-count'},
                "attention_mask": {0: "batch", 1: "seq_length"},
                "overflow_to_sample_mapping": {0: "batch"},
                "word_ids": {0: "batch", 1: "seq_length"},
                "token_embeddings": {0: "sent-count", 1: "max_token_count", 2: "token_embedding_size"}}
output_names = ["token_embeddings"]



if flair.device.type == "cuda":
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
else:
    providers = ["CPUExecutionProvider"]

desired_keys_order = [
    param for param in inspect.signature(embedding.forward).parameters.keys() if param in example_tensors
]
torch.onnx.export(
    embedding,
    (example_tensors,),
    "./onnx_models/embedding2.onnx",
    input_names=desired_keys_order,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=13,
)