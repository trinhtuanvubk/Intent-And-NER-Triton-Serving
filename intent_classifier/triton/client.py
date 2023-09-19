import torch
import tritonhttpclient
import numpy as np
from transformers import Wav2Vec2Processor

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


input_name= ['input_ids', 'attention_mask']
output_name = 'logits'
VERBOSE = False
model_name = 'intent_cls'
url = 'host.docker.internal:8030'
model_version = '1'

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)   


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

input_texts = ["I want to book this hotel", "I want to cancel this meeting right now"]
dummy_model_input = tokenizer(input_texts, padding='max_length', truncation=True, return_tensors="pt")

input_ids = dummy_model_input['input_ids']
attention_mask = dummy_model_input['attention_mask']


triton_input_ids = tritonhttpclient.InferInput(
                                "input_ids",
                                (input_ids.shape[0],  input_ids.shape[1]),
                                'FP32')
triton_input_ids.set_data_from_numpy(np.array(input_ids))

triton_attention_mask = tritonhttpclient.InferInput(
                                "attention_mask",
                                (attention_mask.shape[0],  attention_mask.shape[1]),
                                'FP32')
triton_attention_mask.set_data_from_numpy(np.array(attention_mask))

triton_output = tritonhttpclient.InferRequestedOutput("logits")    


response_triton = triton_client.infer(model_name,
                                     model_version=model_version,
                                     inputs=[triton_input_ids, triton_attention_mask],
                                     outputs=[triton_output])

logits = response_triton.as_numpy('logits')[0]
logits = np.asarray(logits, dtype=np.float32)

# get ids
probabilities = softmax(logits, axis=1)
predicted_ids = np.argmax(probabilities, axis=1)

# map ids to label
predicted_labels = label_encoder.inverse_transform(predicted_ids)
for text, pred in zip(input_texts, predicted_labels):
    print(f"Text: {text} - Predicted: {pred}")