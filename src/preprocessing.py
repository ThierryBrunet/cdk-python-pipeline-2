# -*- coding: utf-8 -*-

"""
@author: yogen.chaudhari@powerschool.com
@proprietary: PowerSchool Group LLC

"""

import os
import pickle
from itertools import islice

# import glob
# import json
import pandas as pd

from any_tree_helper import AnyTreeHelper
from case_preprocessing_helper import CasePreprocessingHelper
from pretrained_embeddings import Embeddings


class Preprocessing:
    def __init__(self, df, framework, file_name, columns_dict, config):
        """

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        framework : TYPE
            DESCRIPTION.
        file_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.df = df
        self.framework = framework.lower()
        self.config = config
        self.embd = Embeddings()
        self.file_name = file_name
        self.columns_dict = columns_dict
        self.embeddings_path = list()

        self.framework_specific_preprocessing()
        self.get_tree_structure()
        self.save_embeddings()

    def framework_specific_preprocessing(self):
        """

        Returns
        -------
        None.

        """
        if self.framework == "case":
            cph = CasePreprocessingHelper(self.df)
            self.child_parent_df = cph.child_parent_df
        elif self.framework == "ab":
            self.child_parent_df = self.df[["GUID", "Parent_GUID"]].rename(
                {"GUID": "Child", "Parent_GUID": "Parent"}, axis=1
            )
        elif self.framework == "ptp":
            pass

    def get_tree_structure(self):
        """


        Returns
        -------
        None.

        """
        self.ls_tree = AnyTreeHelper(
            self.child_parent_df, self.df, self.columns_dict, self.framework
        )
        self.element_level_docs = self.ls_tree.finalElementLevelTextData
        self.all_level_docs = self.ls_tree.finalElementLevelTextDataWeight

    def chunks(self, data, SIZE=500):
        """
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        SIZE : TYPE, optional
            DESCRIPTION. The default is 500.

        Yields
        ------
        dict
            DESCRIPTION.

        """
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k: data[k] for k in islice(it, SIZE)}

    def save_embeddings(self):
        """

        Returns
        -------
        None.

        """
        counter = 0
        for item in self.chunks(self.all_level_docs, 500):
            embeddings_dict = dict()
            temp_path = os.path.abspath(
                self.config["s3_temp_processed_data"]
                + self.file_name
                + "_{}.pickle".format(str(counter))
            )
            with open(temp_path, "wb") as handle:
                self.embeddings_path.append(temp_path)
                for k, v in item.items():
                    try:
                        e1 = self.embd.get_st_embeddings(v["element"])
                        e2 = self.embd.get_st_embeddings(
                            str(v["pre_element"]) + ". " + str(v["element"])
                        )
                        embeddings_dict[k] = {"element": e1, "full": e2}
                    except Exception as e:
                        # print(k,v)
                        print(e)
                        pass
                pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = counter + 1

    #### Recreate grade

    def _grade_helper_(x):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        l, h = x[0], x[1]
        try:

            if l == h:
                return l
            else:
                return l + "_" + h
        except:
            return "NA"

    ### standardize grades

    # CASE
    def caseGradeHelper(x):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if x.lower() == "kg":
            return {0}
        elif x.lower() == "pk":
            return {-1}
        else:
            try:
                return {int(x)}
            except:
                try:
                    x = [
                        0 if item.lower() == "kg" else int(item)
                        for item in x.split(",")
                    ]
                    # return set([int[item] for item in x])
                    return set(x)
                    return x
                except Exception as e:
                    print(e)

    # AB
    def abGradeHelper(x):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if x.lower() == "k":
            return {0}
        elif x.lower() == "na":
            return {"NA"}
        else:
            try:
                x = x.split("_")
                if len(x) == 1:
                    l = x[0]
                    h = x[0]
                else:
                    l, h = x
                if l == "K" or l == "k":
                    l, h = 0, 0
                return set(list(range(int(l), int(h) + 1)))
            except Exception as e:
                print(e)
