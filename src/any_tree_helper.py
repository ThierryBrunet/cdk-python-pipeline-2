# -*- coding: utf-8 -*-

"""
@author: yogen.chaudhari@powerschool.com
@proprietary: PowerSchool Group LLC


"""

import json

from anytree import Node, RenderTree


class AnyTreeHelper:
    def __init__(self, child_parent_data, full_data, columns_dict, framework="NA"):
        """
        Helper class to create and traverse tree data structure from LSM inpute flat csv/tsv files.
        It's built on "anytree" python library

        Keyword Args:
             child_parent_data: preprocessed data which provides child parents relationship of ids
             full_data: the input file
             columns_dict: {"id":"", "parent_id":"", "subject":"", "grade_level":"", "standard":""}
                         This disctionary helps standardizing columns names across multiple source of inputs do that we can reuse code
             framework: Default is "NA" but it can be CASE, AB, PTP etc

        """
        ## Variables initialization
        self.columns_dict = columns_dict
        self.framework = framework.lower()
        self.nodes = {}
        self.roots = []
        self.allElementLevelPaths = []
        self.allElementLevelMappings = {}
        self.finalElementLevelTextData = {}
        self.finalElementLevelTextDataWeight = {}
        self.jsn_data = {}
        self.Id2SmartU_mappings = {}
        self.SmartU2Id_mappings = {}

        ## Function calls
        if self.framework == "case":
            self.get_case_id_mappings(full_data)
        self.get_tree(child_parent_data)
        self.get_full_paths_for_all_leaf_nodes()
        self.get_json(full_data)
        self.get_all_leaf_nodes_text_data()
        self.get_text_data_by_concatenating_tree_path()

    def add_nodes(self, parent, child):
        """
        Helper function to create tree data structure

        Parameters
        ----------
        parent : parent_id.
        child : child id.

        Returns
        -------
        None.

        """
        if parent not in self.nodes:
            self.nodes[parent] = Node(parent)
        if child not in self.nodes:
            self.nodes[child] = Node(child)
        self.nodes[child].parent = self.nodes[parent]

    def get_tree(self, data):
        """
        This functions creates tree structure from child-parent data
        Parameters
        ----------
        data : child_parent_data

        Returns
        -------
        None.

        """
        for parent, child in zip(data["Parent"], data["Child"]):
            self.add_nodes(parent, child)
        self.roots = list(data[~data["Parent"].isin(data["Child"])]["Parent"].unique())

    def get_full_paths_for_all_leaf_nodes(self):
        """
        This function renders through whole tree and store full paths (from root to leaf) for all leaf nodes in dictionary

        Returns
        -------
        None.

        """
        for root in self.roots:
            for pre, _, node in RenderTree(self.nodes[root]):
                if node.is_leaf:
                    self.allElementLevelPaths.append([p.name for p in node.path])
        for l in self.allElementLevelPaths:
            self.allElementLevelMappings[l[-1]] = l

    def get_text_data_by_concatenating_tree_path(self):
        """
        This function stores text data of all leaf nodes as well as
        stores concatinated text data from traversal path from root to respective leaf node.

        Returns
        -------
        None.

        """
        for k, v in self.allElementLevelMappings.items():
            temp_l = list()
            if self.framework == "case":
                element = self.jsn_data[self.SmartU2Id_mappings[v[-1]]][
                    self.columns_dict["standard"]
                ]
            else:
                element = self.jsn_data[v[-1]][self.columns_dict["standard"]]
            exception_set = set()

            for vv in v[:-1]:
                if vv in self.jsn_data:
                    j = self.jsn_data[vv]
                    temp_l.append(j[self.columns_dict["standard"]])
                elif vv != "'ALL-CURR'":
                    exception_set.add(vv)

            pre_element = ". ".join([str(tl) for tl in temp_l])
            self.finalElementLevelTextDataWeight[k] = {
                "element": element,
                "pre_element": pre_element,
            }
        print(exception_set)

    def get_all_leaf_nodes_text_data(self):
        """
        Stores all leaf nodes with text data


        Returns
        -------
        None.

        """
        for k, v in self.allElementLevelMappings.items():
            if self.framework == "case":
                j = self.jsn_data[self.SmartU2Id_mappings[v[-1]]]
            else:
                j = self.jsn_data[v[-1]]
            self.finalElementLevelTextData[k] = j[self.columns_dict["standard"]]

    def get_json(self, full_data):
        """
        This function stores input data in json format so that it can be available
        easily with O(1) complexity while matching the standards

        Parameters
        ----------
        full_data : crosswalk input dataframe.

        Returns
        -------
        None.

        """
        t_full_data = full_data.copy()
        t_full_data.index = t_full_data[self.columns_dict["id"]]
        self.jsn_data = json.loads(t_full_data.to_json(orient="index"))
        del t_full_data

    """
    ########################
    CASE specific processing
    ########################
    """

    def __case_mapping_helper__(self, x):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.Id2SmartU_mappings[x[0]] = x[1]
        self.SmartU2Id_mappings[x[1]] = x[0]

    def get_case_id_mappings(self, df):
        """

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        _ = df[["identifier", "smartLevel_unique"]].apply(
            lambda x: self.__case_mapping_helper__(x), axis=1
        )
