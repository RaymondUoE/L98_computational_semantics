
import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class Featureriser(object):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Cuda is available: {torch.cuda.is_available()}')
    print(f'Device: {device}')
    
    @staticmethod
    def bert_featurerise(edses, sentences):


        print('Loading tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print('Loading BERT...')
        model = BertModel.from_pretrained('bert-base-uncased',
                                        output_hidden_states = True, # returns all hidden-states.
                                        ).to(Featureriser.device)
        model.eval()
        batch_size = 64

        # <CLS> tokens <SEP>
        tokens_list = []
        token_embeddings = []

        for idx in tqdm(range(0, len(sentences), batch_size), total=int(np.ceil(len(sentences)/batch_size))):
            batch = sentences[idx : min(len(sentences), idx+batch_size)]
            
            encoded = tokenizer.batch_encode_plus(batch,max_length=256, padding='max_length', truncation=False)
            for ids in encoded['input_ids']:
                tks = tokenizer.convert_ids_to_tokens(ids)
                tokens_list.append(list(filter(lambda x: x != '[PAD]', tks)))
        
            encoded = {key:torch.LongTensor(value).to(Featureriser.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            lhs = outputs.last_hidden_state
            attention = encoded['attention_mask'].reshape((lhs.size()[0], lhs.size()[1], -1)).expand(-1, -1, 768)
            mask = (attention > 0)[:,:,0]

            # unbatch
            for i, m in enumerate(mask):
                token_embeddings.append(lhs[i,:][m])
        # return tokens_list, token_embeddings
        span_embeddings_list_of_dict = []
        for i, eds in zip(range(len(edses)), edses):
            span_embeddings_list_of_dict.append(Featureriser.get_node_span_embedding(eds, sentences[i], tokens_list[i], token_embeddings[i]))
        return span_embeddings_list_of_dict



    @staticmethod
    def get_node_span_embedding(eds, sentence, tokens, token_embeddings):
        surface_per_node_dict = Featureriser.eds_nodes_to_surface_char_level(eds, sentence)
        span_embeddings = {}
        surface_per_node_dict = Featureriser.match_node_spans_with_token_index(surface_per_node_dict, tokens)
        for n in eds.nodes:
            a, b = surface_per_node_dict[n.id]['token_index']
            embeddings = token_embeddings[a:b]
            span_embeddings[n.id] = torch.mean(embeddings, dim=0).unsqueeze(0)
        return span_embeddings
        

    @staticmethod
    def eds_nodes_to_surface_char_level(eds, sentence):
        surface_per_node = []
        surface_per_node = {}
        for n in eds.nodes:
            if n.lnk.data:
                start = n.lnk.data[0]
                stop = n.lnk.data[1]
                surface_per_node[n.id] = {'surface': sentence[start:stop]}
                surface_per_node[n.id]['start_char'] = int(start)
                surface_per_node[n.id]['stop_char'] = int(stop)
            else:
                surface_per_node[n.id] = {'surface': ''}
        # sort by starting char
        return dict(sorted(surface_per_node.items(), key=lambda x: int(x[1]['start_char'])))

    @staticmethod
    def surface_string_to_token_index(string_to_be_matched, tokens, original_string, a, b, match_started=False):
        if string_to_be_matched == '':
            return a, b
        elif b == len(tokens) & match_started:
            return a, b
        else:
            if match_started:
                start_token = tokens[b]
                if not string_to_be_matched.startswith(start_token): #false match
                    return Featureriser.surface_string_to_token_index(original_string, tokens, original_string, a+1, a+2, False)
                else:
                    return Featureriser.surface_string_to_token_index(string_to_be_matched[len(start_token):].strip(), tokens,original_string, a, b+1, True)

            else:
                start_token = tokens[a]
                if string_to_be_matched.startswith(start_token):
                    return Featureriser.surface_string_to_token_index(string_to_be_matched[len(start_token):].strip(), tokens, original_string, a, b, True)
                else:
                    return Featureriser.surface_string_to_token_index(string_to_be_matched, tokens, original_string, a+1, b+1, False)
    
    @staticmethod
    def match_node_spans_with_token_index(node_spans_dict, tokens):
        # node_spans = [x.lower() for x in node_spans]
        tokens = [x[2:] if x[:2] == '##' else x for x in tokens]
        a = 0
        
        for node_id, info_dict in node_spans_dict.items():
            cur_span = info_dict['surface'].lower()
            start, finish = Featureriser.surface_string_to_token_index(cur_span, tokens, cur_span, a, a+1)
            node_spans_dict[node_id]['token_index'] = (start, finish)
            a = start
        return node_spans_dict

