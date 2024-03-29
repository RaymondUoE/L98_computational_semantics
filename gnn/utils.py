# from delphin.eds._eds import EDS
import copy
import delphin.codecs.eds
import graphviz

from nltk.tree import Tree

def find_df_by_id(idd, df):
    result = df[df['id'] == idd]
    
    if len(result) == 0:
        return None
    else:
        return result.to_dict('records')


def find_tree_by_ids_df(idd, df):
    result = df[df['id'] == idd]
    
    if len(result) == 0:
        return None
    else:
        return Tree.fromstring(result['tree'].values[0])


def eds_vis(eds, sent=None):
    dot = graphviz.Digraph()
    dot.attr(label=sent)
    for n in eds.nodes:
        dot.node(n.id, n.predicate)
    for e in eds.edges:
        dot.edge(e[0], e[2], label=e[1])
    return dot


def arg_number_increase_by_one(arg):
    return 'ARG' + str(int(arg.split('ARG')[-1]) + 1)


def arg_number_decrease_by_one(arg):
    return 'ARG' + str(int(arg.split('ARG')[-1]) - 1)

def eds_from_string(string):
    return delphin.codecs.eds.decode(string)

def eds_to_string(eds):
    return delphin.codecs.eds.encode(eds)

def string_of_list_to_list(string_of_list):
    return string_of_list.strip('[]').replace('\'', '').replace('\"', '').split(', ')

def find_node_ids_edge_targets(eds, semlinks, enhance=False):
    # one eds can have multiple semlinks
    # one semlink can augment multiple edges
    node_index = 0
    semlink_index = 0

    counter_redundant_pb = 0
    node_cannot_be_found = 0

    node_ids = []
    sls = []
    fn_frames = []
    edge_targets = []
    fn_roles = []
    
    enhanced = copy.deepcopy(eds)

    while True:
        # cannot find corresponding verb in eds for a semlink
        if (node_index == len(eds.nodes) and semlink_index < len(semlinks)):
            node_cannot_be_found += 1
            # restart search from next semlink, from first node
            semlink_index += 1
            node_index = 0
            continue

        if semlink_index == len(semlinks):
            break
        cur_sl = semlinks[semlink_index]
        cur_augmentations = string_of_list_to_list(cur_sl['augmentations'])
        # cur_augmentations = cur_sl['augmentations']
        cur_node = eds.nodes[node_index]


        # predicate matches semlink vb form
        if '_'.join(cur_sl['vb_form'].split('-')) in cur_node.predicate:
            node_ids.append(cur_node.id)
            fn_frames.append(cur_sl['fn_frame'])
            sls.append(cur_sl)
            
            if enhance:
                # cur_node.predicate = cur_node.predicate + '-fn.' + cur_sl['fn_frame']
                enhanced.nodes[node_index].predicate = cur_node.predicate + '-fn.' + cur_sl['fn_frame']

            # cur_verb_edge_labels = []
            cur_verb_edge_targets = []
            cur_verb_edge_fn_roles = []
            # looking for edges
            for label in list(cur_node.edges):
                target = cur_node.edges[label]
                augmentations, has_redundant_pb_role = process_augmentations(cur_augmentations)
                if has_redundant_pb_role:
                    counter_redundant_pb += 1
                if arg_number_decrease_by_one(label) in augmentations:
                    cur_verb_edge_targets.append(target)
                    fn_role = augmentations[arg_number_decrease_by_one(label)] 
                    cur_verb_edge_fn_roles.append(fn_role)
                    
                    if enhance and label in enhanced.nodes[node_index].edges: # avoid error from redundany pb role
                        if fn_role != '':
                            new_key = label + '-fn.' + augmentations[arg_number_decrease_by_one(label)]
                            # cur_node.edges[new_key] = cur_node.edges.pop(label)
                            enhanced.nodes[node_index].edges[new_key] = enhanced.nodes[node_index].edges.pop(label)
            

            # after looping through edges
            semlink_index += 1
            edge_targets.append(cur_verb_edge_targets)
            fn_roles.append(cur_verb_edge_fn_roles)

        node_index += 1
    
    return {'node_ids': node_ids, 
            'semlink' : sls,
            'fn_frames': fn_frames, 
            'edge_targets': edge_targets, 
            'fn_roles': fn_roles, 
            'counter_redundant_pb': counter_redundant_pb, 
            'node_cannot_be_found': node_cannot_be_found}, enhanced

def process_augmentations(list_of_augmentations):
    '''return arg0, arg1... and frameNet roles if any'''
    extracted_augs = {}
    has_redundant_pb_role = False

    for aug in list_of_augmentations:
        # remove token
        token_span, rest = aug.split('-', 1)
        # best approximation of existence of pb role, vn role
        if 'ARG' in rest and '=' in rest:
            # extract pb_role
            pb_role, rest = rest.split('=', 1)
            # best approximation of existence of fn role
            if ';' in rest:
                fn_role = rest.split(';')[1]
                # it has redundant pb role possibily due to errornuous annotation
                if pb_role in extracted_augs:
                    has_redundant_pb_role = True
                    # overwrite with better information
                    if extracted_augs[pb_role] == '':
                        extracted_augs[pb_role] = fn_role
                    continue
                else:
                    extracted_augs[pb_role] = fn_role
            # there is no fn role
            else:
                # check not to overwrite better information
                if pb_role not in extracted_augs:
                    extracted_augs[pb_role] = ''
                
        # augmentation doesn't concern pb role, vn role
        else:
            pass
    
    return extracted_augs, has_redundant_pb_role
