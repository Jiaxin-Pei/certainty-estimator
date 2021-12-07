from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .edited_bert import BertModel, BertForSequenceClassification
import torch
from torch import Tensor
import numpy as np
import math


class CertaintyEstimator(object):
    def __init__(self, task = 'sentence-level', cuda = False, use_auth_token=False):
        
        self.task = task
        
        if task == 'sentence-level':
            model_path = 'pedropei/sentence-level-certainty'
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, num_labels=1, output_attentions=False,
                                                         output_hidden_states=False, cache_dir = './model_cache', use_auth_token=use_auth_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1,
                                     output_attentions=False, output_hidden_states=False,cache_dir = './model_cache',use_auth_token=use_auth_token)
        elif task == 'aspect-level':
            model_path = 'pedropei/aspect-level-certainty'
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                 output_hidden_states=False, cache_dir = './model_cache', use_auth_token=use_auth_token)
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3, output_attentions=False, 
                                               output_hidden_states=False, cache_dir = './model_cache', use_auth_token=use_auth_token)
        self.cuda = cuda
        if cuda:
            self.model.cuda()

    def data_iterator(self, train_x, batch_size):
        n_batches = math.ceil(len(train_x) / batch_size)
        for idx in range(n_batches):
            x = train_x[idx * batch_size:(idx + 1) * batch_size]
            yield x

    def padding(self, text, pad, max_len=60):
        return text if len(text) >= max_len else (text + [pad] * (max_len - len(text)))

    def encode_batch(self, text):

        tokenizer = self.tokenizer
        t1 = []
        for line in text:
            t1.append(self.padding(tokenizer.encode(line, add_special_tokens=True, max_length=60, truncation=True),
                              tokenizer.pad_token_id))

        return t1
    
    def predict_sentence_level(self, text, batch_size=128, tqdm=None):
        if type(text) == str:
            text = [text]

        test_iterator = self.data_iterator(text, batch_size)
        all_preds = []
        all_res = []
        if tqdm:
            #print('Please use tqdm to track the progress of model predictions')
            #return None
            for x in tqdm(test_iterator, total=int(len(text) / batch_size)):

                ids = self.encode_batch(x)

                with torch.no_grad():
                    if self.cuda:
                        input_ids = Tensor(ids).cuda().long()
                    else:
                        input_ids = Tensor(ids).long()
                    outputs = self.model(input_ids)

                predicted = outputs[0].cpu().data.numpy()
                all_preds.extend(predicted)

            all_res = np.array(all_preds).flatten()
            return list(all_res)
        else:
            for x in test_iterator:

                ids = self.encode_batch(x)

                with torch.no_grad():
                    if self.cuda:
                        input_ids = Tensor(ids).cuda().long()
                    else:
                        input_ids = Tensor(ids).long()
                    outputs = self.model(input_ids)

                predicted = outputs[0].cpu().data.numpy()
                all_preds.extend(predicted)

            all_res = np.array(all_preds).flatten()
            return list(all_res)

            
    def predict_aspect_level(self, text, get_processed_output, batch_size=128, tqdm=None):
        if type(text) == str:
            text = [text]
            
        test_iterator = self.data_iterator(text, batch_size)
        all_preds = []
        all_res = []
        if tqdm:
            #print('Please use tqdm to track the progress of model predictions')
            #return None
            for x in tqdm(test_iterator, total=int(len(text) / batch_size)):

                ids = self.encode_batch(x)

                with torch.no_grad():
                    if self.cuda:
                        input_ids = Tensor(ids).cuda().long()
                    else:
                        input_ids = Tensor(ids).long()
                    outputs = self.model(input_ids)

                predicted = [y_pred.cpu().data.numpy() for y_pred in outputs[0]]
                all_preds.extend(np.transpose(predicted,(1,0,2)))

            all_res = np.argmax(all_preds, axis=2)
            labels = ['Number','Extent','Probability','Condition','Suggestion','Framing']
            mapping = {0:'Uncertain', 1:'NotPresent', 2:'Certain'}
            res_with_labels = []
            present_aspect_certainty = []
            for res in all_res:
                item = {}
                for i in range(len(labels)):
                    item[labels[i]] = mapping[res[i]]
                res_with_labels.append(item)
                only_present = [it for it in item.items() if it[1] != 'NotPresent']
                present_aspect_certainty.append(only_present)
            if get_processed_output:
                return present_aspect_certainty
            else:
                return all_preds, list(all_res), res_with_labels, present_aspect_certainty
        else:
            for x in test_iterator:

                ids = self.encode_batch(x)

                with torch.no_grad():
                    if self.cuda:
                        input_ids = Tensor(ids).cuda().long()
                    else:
                        input_ids = Tensor(ids).long()
                    outputs = self.model(input_ids)

                predicted = [y_pred.cpu().data.numpy() for y_pred in outputs[0]]
                all_preds.extend(np.transpose(predicted,(1,0,2)))

            all_res = np.argmax(all_preds, axis=2)
            labels = ['Number','Extent','Probability','Condition','Suggestion','Framing']
            mapping = {0:'Uncertain', 1:'NotPresent', 2:'Certain'}
            res_with_labels = []
            present_aspect_certainty = []
            for res in all_res:
                item = {}
                for i in range(len(labels)):
                    item[labels[i]] = mapping[res[i]]
                res_with_labels.append(item)
                only_present = [it for it in item.items() if it[1] != 'NotPresent']
                present_aspect_certainty.append(only_present)
            if get_processed_output:
                return present_aspect_certainty
            else:
                return all_preds, list(all_res), res_with_labels, present_aspect_certainty
            
    def predict(self, text, get_processed_output = True, batch_size=128, tqdm=None):
        if self.task == 'sentence-level':
            return self.predict_sentence_level(text, batch_size, tqdm)
        else:
            return self.predict_aspect_level(text, get_processed_output, batch_size, tqdm)