import argparse
import logging
import sagemaker_containers
import requests

import os
import json
import io
import time
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
# from sentence_transformers import models, losses, SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_tformer(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

    # move input tensors to gpu if applicable
    encoded_input = {k:v.to(model.device) for k,v in encoded_input.items()}
    
    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    nlp_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    nlp_model.to(device)
    model = {'model':nlp_model, 'tokenizer':tokenizer}

#     model = SentenceTransformer(model_dir + '/transformer/')
#     logger.info(model)
    return model

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(serialized_input_data, content_type='text/plain'):
    logger.info('Deserializing the input data.')
    try:
        data = serialized_input_data.decode('utf-8').splitlines()  # assumes each text has no newlines, and is on its on own line
        # print(f'Deserialized data: {data}')  # temp
        return data
    except:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    sentence_embeddings = embed_tformer(model['model'], model['tokenizer'], input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    # print(f'sentence embeddings: {sentence_embeddings}')  # tmp
    response = sentence_embeddings
    # print(f"predict_fn response: {response}") #tmp
    return response

# Serialize the prediction result into the desired response content type
def output_fn(predictions, accept):
    logger.info('Serializing the generated output.')
    if accept == 'application/json':
        output = '\n'.join(json.dumps(prediction) for prediction in predictions.tolist())
        return output
    elif accept == 'application/octet-stream':
        output = predictions.cpu().numpy().astype(np.float32).tobytes()
        return output
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))