# from delphin.eds._eds import EDS
import delphin.codecs.eds
import graphviz

from nltk.tree import Tree
# eds._eds

# class EnhancedEDS(EDS):
#     # def __init__(self,
#     #              top: str = None,
#     #              nodes: Iterable[Node] = None,
#     #              lnk: Lnk = None,
#     #              surface=None,
#     #              identifier=None):
#     #     print('ss')
#     #     if nodes is None:
#     #         nodes = []
#     #     super().__init__(top, list(nodes), lnk, surface, identifier)
    
    
        
#     def search_nodes_by_id (self, idd):
#         for n in self.nodes:
#             if n.id == idd:
#                 return n
#         return None

    
def find_eds_by_ids_df(section_id, doc_id, sentence_id, df):
    result  = df[(df['section_id'] == section_id) & (df['doc_id'] == doc_id) & (df['sentence_id'] == sentence_id)]
    if len(result) == 0:
#         no result found
        return None
    else:
        return eds_from_string(result['eds'].values[0])
    # try:
    #     eds = df[(df['section_id'] == section_id) & (df['doc_id'] == doc_id) & (df['sentence_id'] == sentence_id)]['eds'].values[0]
    #     return delphin.codecs.eds.decode(eds)
    # except:
    #     return None


def find_semlink_by_ids_df(section_id, doc_id, sentence_id, df):
    result = df[(df['section_id'] == section_id) & (df['doc_id'] == doc_id) & (df['sentence_id'] == sentence_id)]
    
    if len(result) == 0:
        return None
    else:
        return result.to_dict('records')


def find_tree_by_ids_df(section_id, doc_id, sentence_id, df):
    result = df[(df['section_id'] == section_id) & (df['doc_id'] == doc_id) & (df['sentence_id'] == sentence_id)]
    
    if len(result) == 0:
        return None
    # else:
    #     return Tree.fromstring(result['tree'].values[0])
    # try:
    #     tree = df[(df['section_id'] == section_id) & (df['doc_id'] == doc_id) & (df['sentence_id'] == sentence_id)]['tree'].values[0]
    #     return Tree.fromstring(tree)
    # except:
    #     return None


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