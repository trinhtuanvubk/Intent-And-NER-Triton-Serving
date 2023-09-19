
import torch
from flair.data import Sentence
from flair.models import SequenceTagger


from flair.embeddings import TransformerWordEmbeddings

model_path = "./checkpoints/best-model.pt"
model = SequenceTagger.load(model_path)
example_sentence = Sentence("This is a sentence.")
longer_sentence = Sentence("This is a way longer sentence to ensure varying lengths work with LSTM.")

reordered_sentences = sorted([example_sentence, longer_sentence], key=len, reverse=True)
# rnn paded need order sentences
tensors = model._prepare_tensors(reordered_sentences)
# 
dynamic_axes={"sentence_tensor" : {0: 'batch_size',
                                  1: 'max_length'},
                "sen_lengths" : {0: 'batch_size'},
                "scores" : {0: 'batch_size',
                            1: 'max_length',
                            2: 'num_tags',
                            3: 'tag_embedding_length'},
                "sen_lengths" : {0: 'batch_size'},
                "params_info": {0: 'num_tags',
                        1: 'tag_embedding_length'}}
print(tensors[0].shape)
print(tensors[1].shape)

print(len(tensors))
torch.onnx.export(
    model,
    tensors,
    "./onnx_models/sequencetagger.onnx",
    input_names=["sentence_tensor", "sen_lengths"],
    output_names=["scores", "sen_lengths", "params_info"],
    dynamic_axes=dynamic_axes,
    opset_version=13,
    verbose=True,
)


