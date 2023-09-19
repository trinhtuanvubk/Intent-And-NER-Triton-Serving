import torch
import os
import onnxruntime
import numpy as np

from flair.data import Sentence, Dictionary, Span, get_spans_from_bio
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from typing import List, Tuple, Optional, Union, Any, Dict, cast
from viterbi import ViterbiDecoder
from flair.training_utils import store_embeddings

# Some default params from debugging
predict_spans = True
force_token_predictions = False
label_name = 'ner'
return_probabilities_for_all_classes = False
embedding_storage_mode = 'none'


# Process Label Encoder
tag_format = "BIO"
tag_dict = Dictionary().load_from_file("./tag_dictionary/tag_dictionary.pkl")
label_dictionary = Dictionary(add_unk = False)
for label in tag_dict.get_items():
    if label == "<unk>":
        continue
    label_dictionary.add_item("O")
    if tag_format == "BIOES":
        label_dictionary.add_item("S-" + label)
        label_dictionary.add_item("B-" + label)
        label_dictionary.add_item("E-" + label)
        label_dictionary.add_item("I-" + label)
    if tag_format == "BIO":
        label_dictionary.add_item("B-" + label)
        label_dictionary.add_item("I-" + label)
if not label_dictionary.start_stop_tags_are_set():
    label_dictionary.set_start_stop_tags()

print(label_dictionary)

# Load Viterbi_decoder
viterbi_decoder = ViterbiDecoder(label_dictionary)



# Load pre-base-embeeding
base_embeddings = TransformerWordEmbeddings(
            "bert-base-uncased",
            layers="-1,-2,-3,-4",
            layer_mean=False,
            allow_long_sentences=True,
            force_device=torch.device("cuda")
        )


# Get input tensors
sentences = [Sentence("to speak to a customer service advisor"), Sentence("I want to book the hotel")]
tensors = base_embeddings.prepare_tensors(sentences)
print(tensors)


# Load Onnx session

embedding_onnx_path = "./onnx_models/embedding.onnx"
embedding_sess = onnxruntime.InferenceSession(embedding_onnx_path, providers=['CUDAExecutionProvider'] )

embedding_inputs = [x.name for x in embedding_sess.get_inputs()]
print(embedding_inputs)
print([x.shape for x in embedding_sess.get_inputs()])
# ['input_ids', 'token_lengths', 'attention_mask', 'overflow_to_sample_mapping', 'word_ids']

embedding_outputs = [x.name for x in embedding_sess.get_outputs()]
print(embedding_outputs)
print([x.shape for x in embedding_sess.get_outputs()])
# ['token_embeddings']

print({k: v.cpu().numpy().shape for k, v in tensors.items()})

# start = time.time()
bert_embedding = embedding_sess.run(embedding_outputs, {k: v.cpu().numpy() for k, v in tensors.items()})[0]

print(bert_embedding.shape)
print(type(bert_embedding))


# --------------------------------------------------------------
tagger_model_path = "./onnx_models/sequencetagger.onnx"

tagger_sess = onnxruntime.InferenceSession(tagger_model_path, providers=['CUDAExecutionProvider'] )


inputs = [x.name for x in tagger_sess.get_inputs()]
print(inputs)
print([x.shape for x in tagger_sess.get_inputs()])

outputs = [x.name for x in tagger_sess.get_outputs()]
print(outputs)
print([x.shape for x in tagger_sess.get_outputs()])

# Get lengths for input_name1
lengths = np.array([len(sentence.tokens) for sentence in sentences])


features = tagger_sess.run(outputs, {inputs[0]: bert_embedding,
                                inputs[1]: lengths})
print(features[0].shape)
 
for sentence in sentences:
    sentence.remove_labels('ner')

return_probabilities_for_all_classes = False
predictions, all_tags = viterbi_decoder.decode(
                        features, return_probabilities_for_all_classes, sentences)
print(predictions)


predict_spans = True
force_token_predictions = False
label_name = 'ner'
for sentence, sentence_predictions in zip(sentences, predictions):
    # BIOES-labels need to be converted to spans
    if predict_spans and not force_token_predictions:
        sentence_tags = [label[0] for label in sentence_predictions]
        sentence_scores = [label[1] for label in sentence_predictions]
        predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
        for predicted_span in predicted_spans:
            span: Span = sentence[predicted_span[0][0] : predicted_span[0][-1] + 1]
            span.add_label(label_name, value=predicted_span[2], score=predicted_span[1])

    # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
    else:
        for token, label in zip(sentence.tokens, sentence_predictions):
            if label[0] in ["O", "_"]:
                continue
            token.add_label(typename=label_name, value=label[0], score=label[1])

# all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
for sentence, sent_all_tags in zip(sentences, all_tags):
    for token, token_all_tags in zip(sentence.tokens, sent_all_tags):
        token.add_tags_proba_dist(label_name, token_all_tags)

embedding_storage_mode = 'none'
store_embeddings(sentences, storage_mode=embedding_storage_mode)

for sentence in sentences:
    for entity in sentence.get_spans('ner'):
        print(f"Text: {sentence}")
        print(f"Entity: {entity} - {entity.tag}")