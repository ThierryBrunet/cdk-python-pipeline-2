import boto3
import numpy as np
import os
import pandas as pd
import json
import networkx as nx
import treelib
import datetime
from botocore.exceptions import ClientError
from itertools import islice
import pickle
import warnings
from time import sleep

# sklearn is optional for lambda execution environment
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    warnings.warn('sklearn failed to import', ImportWarning)


def remove_extra_spaces(text, empty_value=' '):
    """
    Remove extra spaces, but if NaN will use empty_value
    Using ' ' as empty value replaces empty string with single space
    """
    if pd.isna(text):
        return empty_value
    output = ' '.join(text.split())
    return output if output != '' else empty_value


def generate_timestamp(prefix=None):
    """
    Generate timestamp that can be used as a unique name
    """
    prefix = '' if prefix is None else prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f'{prefix}{timestamp}'


def split_s3_path(s3_uri):
    """return bucket, key tuple from s3 uri like 's3://bucket/prefix/file.txt' """
    return s3_uri.replace('s3://','').split('/',1)


def get_parameter(name):
    """Get SSM Parameter value by name"""
    ssm = boto3.client('ssm')
    return ssm.get_parameter(Name=name)['Parameter']['Value']

    
class S3:
    @classmethod
    def exists(cls, obj):
        try:
            obj.load()
            return True
        except ClientError as e:
            if e.response['Error']['Code']==404:
                return False
        else:
            raise(e)
            
    @staticmethod
    def split_s3_path(uri):
        """return bucket, key tuple from s3 uri like 's3://bucket/prefix/file.txt' """
        return uri.replace('s3://','').split('/',1)
    
    @classmethod
    def object_from_uri(cls, uri):
        return boto3.resource('s3').Object(*cls.split_s3_path(uri))
    
    @staticmethod
    def split_uri(uri):
        return uri.replace('s3://','').split('/',1)
    
    @classmethod
    def get_contents(cls, uri, decode=True):
        contents = cls.object_from_uri(uri).get()['Body'].read()
        if decode:
            contents = contents.decode()
        return contents


class CurriculumNode:
    def __init__(self, tree, node_id):
        self.tree = tree
        self.node_id = node_id
        
    def get_children(self):
        return [self.__class__(self.tree, child_node_id) for child_node_id in self.tree.graph[self.node_id]]
    
    def __str__(self):
        return f'<CurriculumNode: {self.node_id}>'
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def meta(self):
        return self.tree.df_idx.loc[self.node_id]
    
    @property
    def description(self):
        return getattr(self.meta, self.tree.description_attr)
    
    @property
    def code(self):
        return getattr(self.meta, self.tree.code_attr)
    
    def get_parent(self):
        in_edges = list(self.tree.graph.in_edges(self.node_id))
        return self.__class__(self.tree, in_edges[0][0]) if in_edges else None
    
    def get_ancestors(self, root_first=False):
        ancestors = []
        parent = self.get_parent()
        while parent is not None:
            ancestors.append(parent)
            parent = parent.get_parent()
        if root_first:
            ancestors.reverse()
        return ancestors
    
    def get_siblings(self, include_self=False):
        parent = self.get_parent()
        children = parent.get_children()
        return children if include_self else [x for x in children if x.node_id!=self.node_id]
    
    def is_leaf(self):
        return len(self.get_children())==0


class CurriculumTree:
    """
    Class representing curriculum network/tree.
    
    Example usage:
    
    tree_sis = utils.CurriculumTree.from_df(
        df_sis,
        id_attr='ID', 
        parent_attr='ParentStandardIdentifier',
        description_attr='Description',
        code_attr='Identifier',
        id_parent_ref_attr='Identifier',
    )
    
    tree_ab = utils.CurriculumTree.from_df(
        df_ab, 
        id_attr='curriculum_id', 
        parent_attr='parent_id',
        code_attr='curriculum_code',
        description_attr='description'
    )
    """
    def __init__(self, df_idx, graph, id_attr, parent_attr, description_attr='description', code_attr=None, id_parent_ref_attr=None):
        self.graph = graph
        self.df_idx = df_idx  # indexed dataframe
        self.description_attr = description_attr
        self.code_attr = id_attr if code_attr is None else code_attr
        self.id_parent_ref_attr = id_attr if id_parent_ref_attr is None else id_parent_ref_attr
    
    @classmethod
    def from_df(cls, df, id_attr, parent_attr, **kwargs):
        """
        Initialize CurriculumTree instance from Pandas Dataframe
        
        Args:
            df: Pandas DataFrame containing standards data
            id_attr: Column name of dataframe containing an id field to use as the node id
            parent_attr: column indicating parent node. can reference id_attr column or another id-like column
            id_parent_ref_attr: id column that the parent_attr column is referencing. defaults to id_attr.
        """
        graph = cls.build_graph_from_df(df, id_attr, parent_attr, id_parent_ref_attr=kwargs.get('id_parent_ref_attr'))
        return cls(df.set_index(id_attr), graph, id_attr, parent_attr, **kwargs)
        
    @classmethod
    def from_cache(cls, df_idx, graph, **kwargs):
        return cls(df_idx, graph, **kwargs)
        
    def node_exists(self, node_id):
        """
        Check if given node id corresponds to a node in the curriculum tree
        """
        return node_id in self.df_idx.index
                
    def get_node(self, node_id):
        return CurriculumNode(self, node_id)
    
    def get_roots(self):
        return [self.get_node(n) for n,d in self.graph.in_degree() if d==0]
    
    def show(self, root_id, depth_limit=1, parent_limit=0, display_fn=None, stdout=True):
        """
        Print tree view
        
        Example usage:
        tree_sis.show(11333, depth_limit=1)
        
        Example output: 
        
        [GR0] Kindergarten
        ├── [ART.K] Kindergarten - Art
        ├── [ELAR.K] Language Arts - Reading - Kindergarten
        ├── [ELAR.PK] Language Arts - Reading - Alt. Kindergarten
        └── [SS.K] Social Studies - Kindergarten
        
        Args:
            root_id: node id of root node
            depth_limit: depth to display (returned tree is depth_limit+1 including root node)
            display_fn is a callable that takes CurriculumNode as input to determine how to print the node
            stdout (bool): True to print to stdout, False to return raw text (that can be saved as file)
        """
        if not display_fn:
            display_fn = lambda node: f'[{node.code}] {node.description}'
        
        tree = treelib.Tree()
        tree.create_node('[*]'+display_fn(self.get_node(root_id)), root_id)
        
        for parent_id, child_id in nx.bfs_edges(self.graph, root_id, depth_limit=depth_limit):
            tree.create_node(display_fn(self.get_node(child_id)), child_id, parent=parent_id)
            
        # show ancestors
        if parent_limit>0:
            ancestors = list(islice(self.get_node(root_id).get_ancestors(root_first=False), parent_limit))
            if len(ancestors)>0:
                ancestors.reverse()
                ancestor_tree = treelib.Tree()
                parent = None
                for ancestor in ancestors:
                    ancestor_tree.create_node(display_fn(ancestor), ancestor.node_id, parent=parent)
                    parent = ancestor.node_id
                ancestor_tree.paste(ancestors[-1].node_id, tree)
                tree = ancestor_tree
        
        return tree.show(stdout=stdout)
        
    def traverse(self, root_id, method='bfs'):
        """
        Generator that iterates over nodes using BFS or DFS, with depth limit
        Args:
            method: 'bfs' for breadth-first search or 'dfs' for depth-first search
        """
        assert method in ['bfs', 'dfs']
        if method == 'bfs':
            iter_edges = nx.bfs_edges
        elif method == 'dfs':
            iter_edges = nx.dfs_edges
        else:
            raise ValueError('method arg not in {dfs, bfs}')
        yield self.get_node(root_id)
        for _parent_id, child_id in iter_edges(self.graph, root_id):
            yield self.get_node(child_id)
            
    @staticmethod
    def find_invalid_parents_idx(df, parent_attr, id_parent_ref_attr):
        """
        Before initializing the tree, check for items that have parents, but the parent id does not exist anywhere
        Returns an index that can be used in df.loc
        """
        
        idx = (df
            .join(df.set_index(id_parent_ref_attr).assign(flag=True).flag, on=parent_attr, how='left')
            .loc[lambda x: (x.flag!=True) & (~x[parent_attr].isna())]
            .index
        )
        node_ids = list(df.loc[idx][parent_attr].unique())
        if len(node_ids)>0:
            print(f'Detected unreferenceable parents: {list(df.loc[idx][parent_attr].unique())}')
        return idx
    
    @classmethod
    def build_graph_from_df(cls, df, id_attr, parent_attr, id_parent_ref_attr=None):
        """
        Build graph from Pandas DataFrame needed to initialize instance
        Convention: edge direction goes from parent to child
        
        Args:
            df: Pandas DataFrame containing standards data
            id_attr: id column to use for node ids
            parent_attr: column indicating parent node. can reference id_attr column or another id-like column
            id_parent_ref_attr: id column that the parent_attr column is referencing. defaults to id_attr.
        """
        id_parent_ref_attr = id_attr if id_parent_ref_attr is None else id_parent_ref_attr
        
        # find rows that have unreferenceable parents and set parent value to None
        df.loc[cls.find_invalid_parents_idx(df, parent_attr, id_parent_ref_attr), parent_attr] = None
        
        graph = nx.DiGraph()
        print("Building graph (nodes) ...")
        for x in df.itertuples():
            graph.add_node(getattr(x, id_attr))
        print("Building graph (edges) ...")
        
        if id_parent_ref_attr!=id_attr:
            # maps from id_parent_ref_attr to other attributes, including parent_attr
            df_reindexed_by_id_parent_ref_attr = df.set_index(id_parent_ref_attr)
            for x in df.loc[~df[parent_attr].isna()].itertuples():
                parent_node_id = df_reindexed_by_id_parent_ref_attr.loc[getattr(x, parent_attr)][id_attr]
                graph.add_edge(parent_node_id, getattr(x, id_attr))
        else:
            for x in df.loc[~df[parent_attr].isna()].itertuples():
                graph.add_edge(getattr(x, parent_attr), getattr(x, id_attr))
        return graph
            
            
class StandardSet:
    """
    Class representing standard set, e.g. SIS, AB or CASE
    """
    @property
    def node_id(self):
        return self.df[self.id_attr]
    
    @property
    def description(self):
        return self.df[self.description_attr]
        
    @property
    def code(self):
        return self.df[self.code_attr]
    
    @property
    def name(self):
        if self.name_attr:
            return self.df[self.name_attr]
        else:
            return self.description
    
    @property
    def tree(self):
        if hasattr(self, '_tree'):
            return self._tree
    
        self._tree = self._generate_tree()
        return self._tree
        
    def _generate_tree(self):
        tree = CurriculumTree.from_df(
            self.df,
            id_attr=self.id_attr,
            parent_attr=self.parent_attr,
            description_attr=self.description_attr,
            code_attr=self.code_attr,
            id_parent_ref_attr=self.id_parent_ref_attr if hasattr(self, 'id_parent_ref_attr') else None
        )
        return tree
    
    def __init__(self, df, *, id_attr, parent_attr, description_attr, code_attr, id_parent_ref_attr=None, name_attr=None):
        self.df = df.copy()
        # column names for id, parent id, description, code columns
        self.id_attr=id_attr
        self.parent_attr=parent_attr
        self.description_attr=description_attr
        self.code_attr=code_attr
        # column that parent_attr column is referencing (could be id or code)
        self.id_parent_ref_attr=id_parent_ref_attr
        # optional name column
        self.name_attr=name_attr
        
        # replace NULL values with nan
        self.df.replace('NULL', np.nan, inplace=True)
                
        # clean up extra spaces in text column
        self.df.loc[:, self.description_attr] = self.df[self.description_attr].apply(remove_extra_spaces)
        if self.name_attr:
            # clean up extra spaces in text column
            self.df.loc[:, self.name_attr] = self.df[self.name_attr].apply(remove_extra_spaces)
        

    @classmethod
    def from_SIS_excel(cls, data_path):
        return cls(
            pd.read_excel(data_path,
                dtype={'ID': str}
            ),
            id_attr='ID',
            parent_attr='ParentStandardIdentifier',
            description_attr='Description',
            code_attr='Identifier',
            id_parent_ref_attr='Identifier',
            name_attr='Name'
        )
        
    @classmethod
    def from_SIS_csv(cls, data_path):
        return cls(
            pd.read_csv(data_path,
                dtype={'ID': str}
            ),
            id_attr='ID',
            parent_attr='parentDistrictStandardId',
            description_attr='Description',
            code_attr='Identifier',
            name_attr='Name'
        )

    @classmethod
    def from_AB_csv(cls, data_path):
        """
        Initialize from AB csv
        """
        return cls(
            pd.read_csv(data_path,
                dtype={'ID': str}
            ),
            id_attr='curriculum_id',
            parent_attr='parent_id',
            code_attr='curriculum_code',
            description_attr='description'
        )
    
    @classmethod
    def from_AB_df(cls, df):
        """
        Initialize from AB csv
        """
        return cls(
            df,
            id_attr='curriculum_id',
            parent_attr='parent_id',
            code_attr='curriculum_code',
            description_attr='description'
        )
    
    @classmethod
    def from_CASE_json(cls, data_path):
        """
        Initialize from single CASE json
        """
        data = json.loads(S3.get_contents(data_path))
        df_items = pd.DataFrame(data['CFItems'])
        df_associations = pd.DataFrame(data['CFAssociations']).loc[lambda x: x.associationType=='isChildOf']
        
        df = df_items.merge(
            (df_associations
             .assign(child_identifier=lambda x: pd.json_normalize(x.originNodeURI)['identifier'])
             .assign(parent_identifier=lambda x: pd.json_normalize(x.destinationNodeURI)['identifier'])
            )[['child_identifier','parent_identifier']],
            left_on='identifier',right_on='child_identifier'
        )
        
        return cls(
            df,
            id_attr='identifier',
            parent_attr='parent_identifier',
            code_attr='humanCodingScheme',
            description_attr='fullStatement'
        )
    
    @property
    def leaf(self):
        """
        return column aligned with df indicating whether node is leaf
        """
        return self.node_id.apply(lambda x: self.tree.get_node(x).is_leaf())

    
class AbMatchIndex:
    
    dim_size = 768
    
    # names of files in index
    annoy_index_file_name = 'annoy/mean-children.ann'  # name of file within index_s3_prefix
    index_metadata_file = 'index.json'
    description_embedding_file = 'description-embedding.npy'
    df_file = 'data.csv'
    mean_children_embedding_file = 'mean-children-embedding.npy'
    ngram_vocab_file = 'ngram-vocab.pkl'
    # settings for CountVectorizer
    ngram_vectorizer_params = dict(
        ngram_range = (2, 2),
        token_pattern=r'(?u)\b\w+\b',  # override default to allow single-char length tokens
    )
    ancestor_embedding_file = 'ancestor-embedding.npy'
    
    def __init__(self, index_s3_prefix, output_filter=None):
        """
        output_filter: filter that only 
        """
        self.index_s3_prefix = index_s3_prefix
        self.standard_set = StandardSet.from_AB_df(self.df)
 
    
    @classmethod
    def build(cls, *, index_s3_location, standard_set, embedding_job_params=None, overwrite=False):
        """
        Build index, by populating several file assets under a common s3 prefix:
            data.csv: text, code, heirarchy info (csv)
            embeddings: cached regular embeddings based on text (npy)
            mean children embeddings: cached embeddings derived from taking mean of child embeddings (npy)
            ngram-vocab: list of ngram tokens
            ngram: cached n-gram term freq embeddings (sparse scipy npz) 
            
        Args:
            standard_set: dataframe of data
            index_s3_location: base s3 url prefix for where index will be located on s3 (s3://...)
        """
        print(f'Building index at {index_s3_location}')
        
        if cls.index_exists(index_s3_location) and not overwrite:
            raise ValueError(f'Index already exists at {index_s3_location}; delete contents manually before continuing')
        
        # process index location
        bucket_name, key_prefix = split_s3_path(index_s3_location)
        bucket = boto3.resource('s3').Bucket(bucket_name)
                
        # upload tabular data
        print('Populating csv data in s3 index')
        standard_set.df.to_csv(os.path.join(index_s3_location, cls.df_file))
    
        # helpers
        embedding_feature_calculator = TextEmbeddingFeatureCalculator(
            standard_set, 
            embedding_job_params=embedding_job_params, 
        )
        
        print('Building description embedding')
        description_embedding = embedding_feature_calculator.description_embedding
        cls._push_numpy_matrix_to_index(description_embedding, index_s3_location, cls.description_embedding_file)

        print('Building mean children embedding')
#         cls._build_mean_children_embedding(index_s3_location, standard_set, description_embedding)
        mean_children_embedding = embedding_feature_calculator.mean_children_embedding
        cls._push_numpy_matrix_to_index(mean_children_embedding, index_s3_location, cls.mean_children_embedding_file)
    
        print('Building ngram vocab')
#         cls._build_ngram(index_s3_location, standard_set)
        ngram_vocab = CodeNgramFeatureCalculator.build_vocab(standard_set)
        cls._push_pickle_to_index(ngram_vocab, index_s3_location, cls.ngram_vocab_file)
    
        print('Building ancestor embedding')
        ancestor_concat_text_embedding = embedding_feature_calculator.ancestor_concat_text_embedding
        cls._push_numpy_matrix_to_index(ancestor_concat_text_embedding, index_s3_location, cls.ancestor_embedding_file)
        
        # helpful metadata to store in index root
        index_data = {
            'version': 1,
            'model_name': embedding_job_params['model_name']
        }
        bucket.Object(os.path.join(key_prefix, cls.index_metadata_file)).put(Body=json.dumps(index_data).encode())
        print("Index built successfully")
    
    @classmethod
    def _push_numpy_matrix_to_index(cls, m, index_s3_location, index_file_name):
        np.save(index_file_name, m)
        bucket_name, index_key_prefix = S3.split_s3_path(index_s3_location)
        boto3.client('s3').upload_file(index_file_name, bucket_name, os.path.join(index_key_prefix, index_file_name))
        
    @classmethod
    def _push_pickle_to_index(cls, obj, index_s3_location, index_file_name):
        bucket_name, index_key_prefix = S3.split_s3_path(index_s3_location)
        with open(cls.ngram_vocab_file, 'wb') as f:
            pickle.dump(obj, f)
        boto3.client('s3').upload_file(index_file_name, bucket_name, os.path.join(index_key_prefix, cls.ngram_vocab_file))
    
    @classmethod
    def index_exists(cls, index_s3_location):
        return S3.exists(S3.object_from_uri(os.path.join(index_s3_location, cls.index_metadata_file))) 
    
    def download_index_file(self, filename):
        """
        Download a file from index location by filename
        """
        bucket_name, key_prefix = S3.split_s3_path(self.index_s3_prefix)
        boto3.client('s3').download_file(bucket_name, os.path.join(key_prefix,filename),filename)
    
    #### cachable features ####
    
    @property
    def df(self):
        if not hasattr(self, '_df'):
            self._df = pd.read_csv(os.path.join(self.index_s3_prefix, self.df_file))
        return self._df
    
    @property
    def description_embedding(self):
        if hasattr(self, '_description_embedding'):
            return self._description_embedding
        self.download_index_file(self.description_embedding_file)
        self._description_embedding = np.load(self.description_embedding_file)
        return self._description_embedding
    
    @property
    def mean_children_embedding(self):
        if hasattr(self, '_mean_children_embedding'):
            return self._mean_children_embedding
        self.download_index_file(self.mean_children_embedding_file)
        self._mean_children_embedding = np.load(self.mean_children_embedding_file)
        return self._mean_children_embedding
    
    @property
    def ngram_vocab(self):
        if hasattr(self, '_ngram_vocab'):
            return self._ngram_vocab
        self.download_index_file(self.ngram_vocab_file)
        with open(self.ngram_vocab_file, 'rb') as f:
            self._ngram_vocab_file = pickle.load(f)
        return self._ngram_vocab_file
    
    @property
    def ngram_embedding(self):
        return CodeNgramFeatureCalculator(self.match_input, match_index.ngram_vocab).ngram_embedding
    
    @property
    def ancestor_concat_text_embedding(self):
        if hasattr(self, '_ancestor_concat_text_embedding'):
            return self._ancestor_concat_text_embedding
        self.download_index_file(self.ancestor_embedding_file)
        self._ancestor_concat_text_embedding = np.load(self.ancestor_embedding_file)
        return self._ancestor_concat_text_embedding
    

class AutomatedMatchCalculator:
    """
    Entrypoint class for doing standards matching task
    """
    
    def __init__(self, match_input, match_index, s3_data_bucket=None, K=5, job_name=None, embedding_job_params=None):
        """
        embedding_job_params is a dict passed to EmbeddingBatchTransformJob.create() and can include kwargs such as:
            job_name
            model_name
            input_prefix
            output_prefix
            data_bucket
            input_key_prefix
            output_key_prefix
            instance_count
            dim_size
        """
        self.match_index = match_index  # MatchOutput class instance
        self.match_input = match_input  # StandardSet class instance
        # could have sagemaker use default bucket if none specifieid
        self.s3_data_bucket = s3_data_bucket
        self.K = K
        self.embedding_job_params = {} if embedding_job_params is None else embedding_job_params
        # use data bucket for embedding job data if no override in embedding job params
        if 'data_bucket' not in self.embedding_job_params:
            self.embedding_job_params['data_bucket'] = self.s3_data_bucket

        self.embedding_feature_calculator = TextEmbeddingFeatureCalculator(
            self.match_input, 
            embedding_job_params=embedding_job_params, 
        )
        
        self.ngram_feature_calculator = CodeNgramFeatureCalculator(self.match_input, match_index.ngram_vocab)

    
    def calculate(self, w1=0.4, w2=0.3, w3=0.3, w4=0.7, w5=0.3):
        """
        Calculate matches by computing and aggregating features
        """
        print('Calculating match results')
        weighted_embedding_input = w1*self.description_embedding + w2*self.mean_children_embedding + w3*self.ancestor_concat_text_embedding
        weighted_embedding_index = w1*self.match_index.description_embedding + w2*self.match_index.mean_children_embedding + w3*self.match_index.ancestor_concat_text_embedding
        print('Combining features ...')
        embedding_similarity = cosine_similarity(weighted_embedding_input, weighted_embedding_index)
        
        ngram_vectorizer = self.ngram_feature_calculator.vectorizer
        input_ngram_embedding = ngram_vectorizer.transform(self.match_input.code.fillna(''))
        index_ngram_embedding = ngram_vectorizer.transform(self.match_index.standard_set.code.fillna(''))
        
        ngram_similarity = cosine_similarity(input_ngram_embedding, index_ngram_embedding)
    
        # weight ngram and embedding distances
        similarity = w4*embedding_similarity + w5*(ngram_similarity)
        print('AutomatedMatchCalculator.calculate() complete')
        
        return similarity

    def find_top_matches(self, similarity):
        print('Identifying top matches')
        top_5_indices = np.flip(np.argsort(similarity, axis=1)[:,-5:], axis=1)  # indices of top 5 elements by score descending for each row
        top_5_scores = np.take_along_axis(similarity, top_5_indices, axis=1)  # top 5 scores per row descending
        
        records = []
        for i, (index_matches, scores) in enumerate(zip(top_5_indices, top_5_scores)):
            q = self.match_input.df.iloc[i]
            matches = [self.match_index.df.iloc[j] for j in index_matches]

            record = {
                'input_id': q[self.match_input.id_attr],
                'input_code': q[self.match_input.code_attr],
                'input_name': q[self.match_input.name_attr] if self.match_input.name_attr else None,
                'input_description': q[self.match_input.description_attr],
                
                'match_id': matches[0][self.match_index.standard_set.id_attr],
                'match_code': matches[0][self.match_index.standard_set.code_attr],
                'match_text': matches[0][self.match_index.standard_set.description_attr],
                'match_score': scores[0],

                'matches_id': [m[self.match_index.standard_set.id_attr] for m in matches],
                'matches_code': [m[self.match_index.standard_set.code_attr] for m in matches],
                'matches_text': [m[self.match_index.standard_set.description_attr] for m in matches],
                'matches_score': scores.tolist(),
            }
            records.append(record)

        df_report = pd.DataFrame(records)
        return df_report
    
    #### cacheable features for match input ####
    
    @property
    def description_embedding(self):
        """
        Get or calculate description embedding
        """
        return self.embedding_feature_calculator.description_embedding
    
    @property
    def mean_children_embedding(self):
        return self.embedding_feature_calculator.mean_children_embedding

    @property
    def ancestor_concat_text_embedding(self):
        return self.embedding_feature_calculator.ancestor_concat_text_embedding
    
    @property
    def ngram_embedding(self):
        return self.ngram_feature_calculator.ngram_embedding


class EmbeddingBatchTransformJob:
    """
    Class for working with Batch Transform jobs that generate embeddings
    """
    def __init__(self, job_name):
        self.job_name = job_name
        self.job = boto3.client('sagemaker').describe_transform_job(TransformJobName=job_name)
        tags = self.get_tag_attributes()
        # knowing num items and dim size is helpful when initializing dims for local array
        self.num_items = int(tags['num_items']) # or infer from transform input
        self.dim_size = int(tags['dim_size'])
        self.embedding_format = tags['embedding_format']
        
    @classmethod
    def create(cls, input_data, *, model_name, dim_size, 
        job_name=None, job_name_prefix=None, job_name_suffix=None,
        input_prefix=None, output_prefix=None, 
        data_bucket=None, input_key_prefix='data/batch-input', output_key_prefix='data/batch-output', 
        num_partitions=None, 
        instance_count=15, instance_type='ml.m5.large', max_concurrent_transforms=None,
        embedding_format='binary',  
    ):
        """
        Create a new batch transform job from list of texts
        
        Specify input_prefix / output_prefix (full s3 object url prefixes including bucket, but not last folder level which will be job specific and created dynamically)
        or combination of data_bucket (bucket name) + input_key_prefix/output_key_prefix (prefix without bucket)
        
        Args:
            input_data: list of texts to run transform on
            model_name: name of sagemaker model to use
            job_name: job name to use - must not already exist
            input_prefix: s3 url prefix of input data
            output_prefix: s3 url prefix of output data
            data_bucket: s3 bucket where input/output data is located (required if using input_key_prefix/output_key_prefix instead of input_prefix/output_prefix)
            input_key_prefix: s3 key prefix of input data within data bucket
            output_key_prefix: s3 key prefix of output data within data bucket
            job_name_prefix: prefix to use when creating job name. Job name created with pattern {prefix}{timestamp}{suffix}
            job_name_suffix: suffix to use when creating job name. Job name created with pattern {prefix}{timestamp}{suffix}
        """
        if job_name is not None:
            if cls.job_exists(job_name):
                raise ValueError('Tried creating job, but job name already exists')
        else:
            job_name = cls.generate_job_name(prefix=job_name_prefix, suffix=job_name_suffix)
        
#         print(f'Preparing new transform job: {job_name}')
        
        #validate input assuming empty text is not a valid model input
        if any((len(x)==0 for x in input_data)):
            raise ValueError('Empty text record found in batch transform input')
            
        input_prefix = f's3://{os.path.join(data_bucket, input_key_prefix, job_name)}' if input_prefix is None else input_prefix
        output_prefix = f's3://{os.path.join(data_bucket, output_key_prefix, job_name)}' if output_prefix is None else output_prefix
        
        # calculate total number of items
        num_items = len(input_data)
        
        # number of partitions defaults to number of instances
        num_partitions = instance_count if num_partitions is None else num_partitions
            
        # upload input texts to be used in batch transform job to s3 in partitions
        cls.split_and_upload_list(input_data, input_prefix, num_partitions=num_partitions)
        
        # determine embedding output format:
        # binary: binary flattened array of multiple embeddings. Can use numpy.frombuffer() to deserialize then reshape
        # text: json-formatted list of embeddings. About 5x size
        
        if embedding_format == 'binary':
            assemble_with = 'None'
            accept = 'application/octet-stream'
        elif embedding_format == 'text':
            assemble_with = 'Line'
            accept = 'application/json'
        
        # api call params
        params = dict(
            TransformJobName=job_name,
            ModelName=model_name,  # name of existing model
            BatchStrategy='SingleRecord',
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_prefix.rstrip('/'),
                    }
                },
                'ContentType': 'text/plain',
                'CompressionType': 'None',
                'SplitType': 'Line',
            },
            TransformOutput={
                'S3OutputPath': output_prefix,
                'AssembleWith': assemble_with,
                'Accept': accept,
            },
            TransformResources={
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
            },
            Tags=[
                {'Key': 'num_items', 'Value': str(num_items)},
                {'Key': 'dim_size', 'Value': str(dim_size)},
                {'Key': 'embedding_format', 'Value': embedding_format}
            ],
        )
        if max_concurrent_transforms:
            params['MaxConcurrentTransforms'] = max_concurrent_transforms
        
        # api call to create transform job
        boto3.client('sagemaker').create_transform_job(**params)
        
        print(f'Created new transform job: {job_name}')
        
        # initialize class without waiting for job to finish
        # sleep(10)  # do wait short period before initializing so that tag attributes will be available
        return cls(job_name)
    
    def get_tag_attributes(self):
        """
        Get dictionary of tags on transform job
        Assumes job has already been created and self.job contains job info from DescribeTransformJob
        """
        arn = self.job['TransformJobArn']
        print("Retrieving tag attributes...")
        EXPECTED_TAG_NAMES = ['num_items', 'dim_size', 'embedding_format']
        MAX_WAIT_TIME = 120
        WAIT_INTERVAL = 5
        elapsed_time = 0
        # waits up to MAX_WAIT_TIME for tags to be available
        # (tags not guaranteed to be available immediately after job initiation)
        tags = {}
        while not all([tag_name in tags for tag_name in EXPECTED_TAG_NAMES]):
            sleep(WAIT_INTERVAL)
            elapsed_time += WAIT_INTERVAL
            if elapsed_time >= MAX_WAIT_TIME:
                raise ValueError('Tags not retrievable before max wait time')
            tags = {tag['Key']: tag['Value'] for tag in boto3.client('sagemaker').list_tags(ResourceArn=arn)['Tags']}

        return tags
        
    @classmethod
    def from_job_name(cls, job_name):
        """
        Initialize class instance based on an existing Batch Transform job
        """
        if cls.job_exists(job_name):
            return cls(job_name)
        else:
            raise ValueError(f'Could not find batch transform job: {job_name}')
    
    def wait_for_job(self):
        """
        Block until Batch Transform job is in a completed or stopped state
        """
        print('Waiting for Batch Transform job to finish...')
        waiter = boto3.client('sagemaker').get_waiter('transform_job_completed_or_stopped')
        waiter.wait(
            TransformJobName=self.job_name,
            WaiterConfig={
                'Delay': 10,
                'MaxAttempts': 1000
            }
        )
        print('Batch Transform job complete')
    
    @staticmethod
    def job_exists(job_name):
        if job_name is None:
            return False
        try:
            boto3.client('sagemaker').describe_transform_job(TransformJobName=job_name)
            return True
        except ClientError as e:
            if e.response['Error']['Message'].startswith('Could not find requested job'):
                return False
            else:
                raise(e)
    
    @staticmethod
    def generate_job_name(prefix=None, suffix=None):
        """
        Generate a unique job name based on current timestamp
        """
        prefix = '' if prefix is None else prefix
        suffix = '' if suffix is None else suffix
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f'{prefix}{timestamp}{suffix}'
    
    @property
    def texts_prefix(self):
        return self.job['TransformInput']['DataSource']['S3DataSource']['S3Uri']
    
    @property
    def embeddings_prefix(self):
        return self.job['TransformOutput']['S3OutputPath']
    
    @staticmethod
    def split_and_upload_list(input_list, input_prefix, num_partitions=15):
        """
        Split a list into <num_partitions> parts and upload each as a text file on s3
        with each element of list delimited by newline character
        """
        assert num_partitions < 100
        bucket_name, prefix = split_s3_path(input_prefix)
        bucket = boto3.resource('s3').Bucket(bucket_name)

        for i, partition in enumerate(np.array_split(input_list, num_partitions)):
            key = os.path.join(prefix, f'input_{i:02}.txt')
            bucket.Object(key).put(Body='\n'.join(partition).encode())
    
    @property
    def embedding_matrix(self):
        """
        Load embeddings from text partitions in s3 to array in memory
        """
        if hasattr(self, '_embedding_matrix'):
            return self._embedding_matrix
        
        # job may have been started recently, so wait for it to finish
        self.wait_for_job()
        
        # initialize space
        self._embedding_matrix = np.empty((self.num_items, self.dim_size))
        
        if self.embedding_format == 'text':
            # load rows from output files and store in array
    #         print('Loading embeddings from transform output into local array ...')
            for i, line in enumerate(self.iter_lines_from_prefix(self.embeddings_prefix)):
                self._embedding_matrix[i] = json.loads(line)
                
        elif self.embedding_format == 'binary':
            bucket_name, key_prefix = split_s3_path(self.embeddings_prefix)
            bucket = boto3.resource('s3').Bucket(bucket_name)
            s3_objs = bucket.objects.filter(Prefix=key_prefix)
            download_array = lambda obj: np.frombuffer(obj.get()['Body'].read(), dtype=np.float32)
            arrays = [download_array(obj) for obj in s3_objs]
            tmp = np.empty(self.num_items*self.dim_size)
            np.concatenate(arrays, axis=0, out=tmp)
            self._embedding_matrix = tmp.reshape((self.num_items, self.dim_size))
        
        return self._embedding_matrix
    
    @staticmethod
    def iter_lines_from_prefix(location):
        """Iterate through lines of files under s3 prefix"""
        bucket_name, prefix = split_s3_path(location)
        bucket = boto3.resource('s3').Bucket(bucket_name)
        src_files = sorted(bucket.objects.filter(Prefix=prefix), key=lambda x: x.key)
        for obj in src_files:
            for line in obj.get()['Body'].read().decode().splitlines():
                yield line


class MeanChildrenContextVectorCalculator:
    """
    class to help calculate context vectors by taking mean of children vectors
    
    Args
        num_items: number of total nodes
        dim_size: dimension of embeddings
        tree: CurriculumTree
        node_ids: iterable of node ids associated with embeddings
        embeddings: embedding matrix with dims (num_items, dim_size)
    """
    def __init__(self, tree, node_ids, base_embeddings):
#         print('Calculating mean children feature ...')
        self.tree = tree
        num_items = len(base_embeddings)
        self.embeddings = np.zeros(base_embeddings.shape)  # pre-allocate np array to store output embeddings
        self.cache_index = {}  # mapping from node_id (non-contiguous integer or string) to 0-index on self.embeddings
        
#         print('Loading embeddings into initial cache ...')
        # populate cache from node/embedding iterators
        for i, (node_id, base_embedding) in enumerate(zip(node_ids, base_embeddings)):
            node = tree.get_node(node_id)
            self.cache_index[node_id] = i
            if node.is_leaf():
                self.embeddings[i] = base_embedding
        
    def set_embedding(self, node_id, embedding):
        cache_idx = self.cache_index[node_id]
        self.embeddings[cache_idx] = embedding
    
    def get_embedding(self, node_id):
        return self.embeddings[self.cache_index[node_id]]
    
    def calculate_context_vector(self, node):
        if node.is_leaf():
            embedding = self.get_embedding(node.node_id)
        else:
            embedding = np.mean([self.calculate_context_vector(child) for child in node.get_children()], axis=0)
            
        self.set_embedding(node.node_id, embedding)
        return embedding
    
    def calculate(self):
        root_nodes = self.tree.get_roots()
        
        for root_node in root_nodes:
            self.calculate_context_vector(root_node)
        # after this cache should reflect calculated embeddings
        return self.embeddings
        

class AncestorTextTreeFeatureCalculator:
    """Helper class to calculate text on which ancestor text embedding feature is based on"""
    def __init__(self, tree, empty_value=' '):
        # for each root
        # walk downwards through tree, accumulating text. set accumulated text as feature for each child node
        # output is a dict of {node_id: feature}
        self.tree = tree
        self.results = {}
        for root_node in self.tree.get_roots():
            self.results[root_node.node_id] = ''
            self._calculate_for_descendants(root_node)
        # second pass to enforce empty value default
        if empty_value != '': 
            for node_id in self.results:
                if self.results[node_id] == '':
                    self.results[node_id] = empty_value
    
    def _calculate_for_descendants(self, node): 
        for child in node.get_children():
            self.results[child.node_id] = (self.results[node.node_id] + ' ' + node.description).strip()
            self._calculate_for_descendants(child)
    

class TextEmbeddingFeatureCalculator:
    """
    Calculates sentence transform text embedding based features for standard set
    Includes description, mean children, and ancestor approach
    """
    dim_size = 768
    def __init__(self, standard_set, embedding_job_params=None):
        self.standard_set = standard_set

        self.embedding_job_params = embedding_job_params if embedding_job_params is not None else {}
        self.embedding_job_params['dim_size'] = self.embedding_job_params.get('dim_size', self.dim_size)
        
    @property
    def description_embedding(self):
        if not hasattr(self, '_description_embedding'):
            self._run_combined_batch_transform()
        return self._description_embedding

    @property
    def name_embedding(self):
        if not hasattr(self, '_name_embedding'):
            self._run_combined_batch_transform()
        return self._name_embedding
    
    @property
    def mean_children_embedding(self):
        if hasattr(self, '_mean_children_embedding'):
            return self._mean_children_embedding
        
        print('Calculating mean children embedding features ...')
        cvc = MeanChildrenContextVectorCalculator(
            tree=self.standard_set.tree,
            node_ids=self.standard_set.node_id,
            base_embeddings=self.description_embedding  # use base embedding as basis for aggregation
        )
        cvc.calculate()
        self._mean_children_embedding = cvc.embeddings
        return self._mean_children_embedding
        
    @property
    def ancestor_concat_text_embedding(self):
        if not hasattr(self, '_ancestor_concat_text_embedding'):
            self._run_combined_batch_transform()
        return self._ancestor_concat_text_embedding
        
    def _embedding_batch_transform(self, input_texts, embedding_job_params=None):
        """
        Use Batch Transform job to generate embeddings for input_texts
        Can use results of previous job if specified via job_name in embedding_job_params dict
        """
        if EmbeddingBatchTransformJob.job_exists(embedding_job_params.get('job_name')):
            print("Existing job found, loading results from output")
            job = EmbeddingBatchTransformJob.from_job_name(
                embedding_job_params['job_name'],
            )
        else:
            print("Existing job not found, kicking off new job")
            job = EmbeddingBatchTransformJob.create(input_texts, **embedding_job_params)
        
        return job.embedding_matrix
        
    def _run_combined_batch_transform(self):
        """
        Construct a transform job that gets embeddings with the same model for a couple features at once:
        - description field
        - name field
        - combined ancestor descriptions
        Set results as self._combined_embedding
        """
        if EmbeddingBatchTransformJob.job_exists(self.embedding_job_params.get('job_name')):
            print("Existing job found, loading results from output")
            job = EmbeddingBatchTransformJob.from_job_name(self.embedding_job_params['job_name'])
        else:
            # get description texts
            description_texts = self.standard_set.description.fillna(self.standard_set.name).tolist()
            # get name texts
            name_texts = self.standard_set.name.fillna(self.standard_set.description).tolist()
            # get ancestor texts
            # join converts node dict to list aligned with dataframe order
            ancestor_texts = self.standard_set.df.join(
                pd.Series(AncestorTextTreeFeatureCalculator(self.standard_set.tree).results, name='ancestor_text'),
                on=self.standard_set.id_attr
            ).ancestor_text.apply(remove_extra_spaces).tolist()

            # execute batch transform
            input_texts = description_texts + name_texts + ancestor_texts
            job = EmbeddingBatchTransformJob.create(input_texts, **self.embedding_job_params)
        
        combined_embedding = job.embedding_matrix
        
        # cache features separately
        num_items = len(self.standard_set.df)
        self._description_embedding = combined_embedding[:num_items]
        self._name_embedding = combined_embedding[num_items:num_items*2]
        self._ancestor_concat_text_embedding = combined_embedding[num_items*2:num_items*3]
        
        
class CodeNgramFeatureCalculator:
    
    # settings for CountVectorizer
    ngram_vectorizer_params = dict(
        ngram_range = (2, 2),
        token_pattern=r'(?u)\b\w+\b',  # override default to allow single-char length tokens
    )
    
    def __init__(self, standard_set, vocab):
        self.standard_set = standard_set
        self.vocab = vocab
        self.vectorizer = CountVectorizer(vocabulary=self.vocab, **self.ngram_vectorizer_params)
    
    def ngram_embedding(self):
        if hasattr(self, '_ngram_embedding'):
            return self._ngram_embedding
        self._ngram_embedding = self.vectorizer.transform(self.standard_set.code)
        return self._ngram_embedding
    
    @classmethod
    def build_vocab(cls, standard_set):
        v = CountVectorizer(**cls.ngram_vectorizer_params)
        v.fit(standard_set.code)
        return v.vocabulary_

    
class GroundTruthManifest:
    
    def __init__(self, data, manifest_type='output'):
        self.data = data
        self.manifest_type = manifest_type  # input or output
    
    @classmethod
    def from_s3_uri(cls, uri, manifest_type='output'):
        content = S3.get_contents(uri)
        data = [json.loads(line) for line in content.splitlines()]
        return cls(data, manifest_type)
        
    @classmethod
    def from_job_name(cls, labeling_job_name, manifest_type='output'):
        labeling_job = boto3.client('sagemaker').describe_labeling_job(LabelingJobName=labeling_job_name)
        
        if manifest_type=='output':
            uri = labeling_job['LabelingJobOutput'].get('OutputDatasetS3Uri')
            if not uri:
                raise ValueError(f'Output manifest not available for labeling job {labeling_job_name}')
        elif manifest_type=='input':
            uri = labeling_job['InputConfig']['DataSource']['S3DataSource']['ManifestS3Uri']
        else:
            raise ValueError('manifest_type not recognized')

        return cls.from_s3_uri(uri, manifest_type=manifest_type)
        
        
class CustomGroundTruthManifest(GroundTruthManifest):
    
    label_attribute_name = 'human-label'
    
    @staticmethod
    def _display_tree(standard_set, node_id, max_len=50000):
        """
        Print text-based representation of tree, with <b> tags around the target node_id
        """
        display_fn = lambda node: f'{"<b>" if node.node_id==node_id else ""}[{node.code}] {node.description}{"</b>" if node_id==node_id else ""}'
        display_text = standard_set.tree.show(node_id, depth_limit=1, parent_limit=8, display_fn=display_fn, stdout=False)
        if len(display_text)>max_len:
            display_text = display_text[:max_len]
        return display_text

    @classmethod
    def input_manifest_from_match_results(cls, match_input, match_index, df_match_report):
        """
        Initialize and create input manifest from match dataframe results
        Match report can be filtered before passing in to sample results for labeling
        """
        df_formatted = (df_match_report.copy()
            .assign(input_id=df_match_report.input_id.apply(remove_extra_spaces, empty_value='n/a'))
            .assign(input_name=df_match_report.input_name.apply(remove_extra_spaces, empty_value='n/a'))
        )
        manifest_lines = [
            {
                'source': str(x.input_id),
                'source-metadata': {
                    'reference':{
                        'id':str(x.input_id), 
                        'code':str(x.input_code), 
                        'name':str(x.input_name),
                        'description': str(x.input_description),
                        'tree':cls._display_tree(match_input, x.input_id)
                    },
                    'matches': [{
                        'id':str(m[0]), 
                        'code':str(m[1]), 
                        'text':str(m[2]),
                        'tree':cls._display_tree(match_index, m[0]),
                    } for m in zip(x.matches_id, x.matches_code, x.matches_text)]
                }
            }
            for x in df_formatted.itertuples()
        ]
        return cls(manifest_lines, manifest_type='input')
        
    def upload(self, target_uri):
        # convert parsed data object to raw text to write
        content = '\n'.join([json.dumps(line) for line in self.data])
        return boto3.resource('s3').Object(*S3.split_uri(target_uri)).put(Body=content)
    
    def parsed(self, include_text=False):
        """
        Return list of dicts containing parsed inputs/outputs (can be easily converted to dataframe)
        Specific to custom task template, expects keys [source-metadata][reference][id]
        
        Return:
        [
            {'input_id':..., 'matches_id': [...], 'comment': '', ...},
            ...
        ]
        """
        if include_text:
            input_fields = (
                {
                    'input_id':x['source-metadata']['reference']['id'], 
                    'code':x['source-metadata']['reference']['code'],
                    'name':x['source-metadata']['reference']['name'],
                    'description':x['source-metadata']['reference']['description']
                }
                for x in self.data
            )
        else:
            input_fields = ({'input_id':x['source-metadata']['reference']['id']} for x in self.data)
        return [
            {**input_fields, **label_fields} 
            for input_fields, label_fields in zip(input_fields, self.iter_first_annotations(parse=True))
        ]
    
    def iter_first_annotations(self, parse=False):
        """
        Iterate through annotations, only considering first annotation for each labeled object
        """
        for obj in self.data:
            annotations = obj[self.label_attribute_name]['annotationsFromAllWorkers']
            if len(annotations)<1:
                yield None
            else:
                first_annotation = json.loads(annotations[0]['annotationData']['content'])
                if parse:
                    first_annotation = self.parse_annotation(first_annotation)
                yield first_annotation
                
    @staticmethod
    def parse_annotation(data):
        """
        Specific to custom task template
        Parse content of a human annotation that looks like this:
          {
            "comment": "adf",
            "match__1__123456": {
              "on": true
            },
            "match__2__123457": {
              "on": false
            },
            "match__3__123458": {
              "on": true
            },
            "no-match-expected": {
              "on": true
            },
            "write-in-id": "123"
          }

        where "comment" and "write-in-id" are optional keys
          
        Desired output is
        {
            matches_rank:[2], 
            matches_id: [2066156], 
            comment: "This is an additional comment", 
            "write-in-id": "123456"
        }
        where "comment" and "write-in-id" are optional keys
        """
        output = {}
        output['comment'] = data.get('comment')
        output['write_in_id'] = data.get('write-in-id')
        output['no_match_expected'] = data['no-match-expected']['on']
        output['matches_rank'] = []
        output['matches_id'] = []
        for key in data:
            if key.startswith('match__'):
                if data[key]['on'] == True:
                    _, rank, match_id = key.split('__')
                    output['matches_rank'].append(int(rank))
                    output['matches_id'].append(match_id)  # not converting from string

        return output


#### not currently used ####

class PropagatorCalculator:
    """Abstract class for methods that propagate results downstream"""
    def propagate_results(self):
        print('propagating results downstream')
        self.results = self.root_results.copy()
        root_nodes = self.tree.get_roots()
        for root_node in root_nodes:
            for parent_node_id, child_node_id in nx.dfs_edges(self.tree.graph, root_node.node_id):
                parent_result = self.results.get(parent_node_id)
                if parent_result is not None:
                    self.results[child_node_id] = parent_result
        return self.results
            
    def calculate(self):
        """
        Apply labeling and populate self.results
        """
        self.calculate_root_results()
        self.propagate_results()
        return self.results


class GradeClassifierCalculator(PropagatorCalculator):
    """
    Apply grade classification, then label rest of relevant downstream nodes according to tree structure
    """
    
    GRADE_INDICATOR_TOKENS = set([
        'K-2','3-4','8-10','11-12','4-7','5-7',
        '9-12','11-12','9-10','6-8','3-5','8-12',
        '4-5','2-3','K-3',
        '912','K2','35','68',
        'PK-4','6-12','K-5',
        'K','P','PK'
    ])
    NUMERIC_TOKENS = set([
        '1','2','3','4','5','6','7','8','9','10','11','12'
    ])
    GRADE_TOKENS = GRADE_INDICATOR_TOKENS.union(NUMERIC_TOKENS)

    @classmethod
    def token_list_score(cls, tokens):
        score = (
            0.5*sum(t in cls.GRADE_INDICATOR_TOKENS for t in tokens)
            + 0.5*sum(t in cls.GRADE_TOKENS for t in tokens))/len(tokens)
        return score

    @classmethod
    def grade_class(cls, token):
        MAPPING = {
            'K2': 'K-2',
            '912': '9-12',
            '35': '3-5',
            '68': '6-8',
        }
        if token not in cls.GRADE_TOKENS:
#             print(f'Unmatched grade token: {token}')
            return None
        else:
            return MAPPING[token] if token in MAPPING else token

    @classmethod
    def classify_children_grade(cls, node):
        """
        Return list of tuples (node, class) where class is the predicted grade level of the node
        """
        children = node.get_children()
        children_codes = [child.code for child in children]

        # assumes grade could be found in 3rd or 4th level of code, but unknown which level
        tokens_3, tokens_4 = [], []
        for child in children:
            tokens = child.code.split('.')
            if len(tokens)>=3:
                tokens_3.append(tokens[2])
            if len(tokens)>=4:
                tokens_4.append(tokens[3])

        score_3 = cls.token_list_score(tokens_3)
        score_4 = cls.token_list_score(tokens_4)
        if score_3 <= 0 and score_4 <= 0:
            return [None]*len(children)

        if score_3 >= score_4:
            tokens_for_prediction = tokens_3
        else:
            tokens_for_prediction = tokens_4
        predictions = [(child.node_id, cls.grade_class(t)) for child, t in zip(children, tokens_for_prediction)]

#         print(f'codes: {children_codes}')
#         print(f'predictions: {predictions}')
        return predictions
    
    def __init__(self, tree, parent_node_ids):
        """
        tree: CurriculumTree
        parent_node_ids: list of node ids, classifier will be applied only to children of these nodes
        """
        self.results = {}  # node_id -> class
        self.root_results = {}  # node_id -> class, results before propagating
        self.parent_node_ids = parent_node_ids
        self.tree = tree
        
    def calculate_root_results(self):
        for parent_node_id in self.parent_node_ids:
            for child_node_id, prediction in self.classify_children_grade(self.tree.get_node(parent_node_id)):
                self.root_results[child_node_id] = prediction
        return self.root_results
            
        
class SubjectClassifierCalculator(PropagatorCalculator):
    @classmethod
    def classify_subject_description(cls, raw_s):
        s = str(raw_s).lower()
        for c in [':','-','/',',','.']:
            s = s.replace(c,' ')
        words = s.split()
        s = ' '.join(w.strip() for w in words)
    #     print(words)
        if any([x in words for x in ['math','mathematics','algebra','geometry','calculus','trigonometry','precalculus','statistics','mathematical']]):
            return 'math'
        elif 'computer science' in s:
            return 'computer science'
        elif 'world languages' in s or any([x in words for x in ['french','spanish']]):
            return 'languages'
        elif 'financial literacy' in s or any([x in words for x in ['financial','economics','accounting']]):
            return 'economics'
        elif 'language arts' in s or any([x in words for x in ['english', 'language','literature','writing']]):
            return 'english'
        elif 'social science' in s or 'social studies' in s or any([x in words for x in ['geography','politics','history']]):
            return 'social studies'
        elif any([x in words for x in ['science','chemistry','physics','biology','scientific','psychology','electronics','engineering','environment']]):
            return 'science'
        elif 'physical education' in s or any([x in words for x in ['physical', 'pe']]):
            return 'physical education'
        elif 'health education' in s:
            return 'health education'
        elif 'music' in s:
            return 'music'
        elif any([x in words for x in ['art','arts','drama','theater','theatre','dance']]):
            return 'art'
        elif any([x in words for x in ['workforce','consumer','technology','career','technologies','business','automotive','technological']]):
            return 'career'
        elif any([x in words for x in ['communication','character','inquiry','organization','collaboration']]):
            return 'study skills'
        else:
            return None
        
    @classmethod
    def classify_subject_node(cls, node):
        return cls.classify_subject_description(node.description)
    
    def __init__(self, tree, node_ids):
        """
        tree: CurriculumTree
        node_ids: list of node ids to apply classification to, before propagating results
        """
        self.tree = tree
        self.node_ids = node_ids
        self.root_results = {}
        self.results = {}
        
    def calculate_root_results(self):
        for node_id in self.node_ids:
            prediction = self.classify_subject_description(self.tree.get_node(node_id).description)
            self.root_results[node_id] = prediction
        return self.root_results
