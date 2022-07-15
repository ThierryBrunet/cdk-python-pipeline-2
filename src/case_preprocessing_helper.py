# -*- coding: utf-8 -*-

"""
@author: yogen.chaudhari@powerschool.com
@proprietary: PowerSchool Group LLC

"""


import pandas as pd


class CasePreprocessingHelper:
    def __init__(self, df):
        """
        Constructor method

        Parameters
        ----------
        df : CASE standards dataframe

        Returns
        -------
        None.
        But stores child_parent_df in object which is useful in creating tree structure

        """
        self.child_parent_list = list()
        self.child_parent_df = None
        self.get_child_parent_df(df)

    def smart_level_recode(self, x):
        """
        This function recodes "SmartLevel" column from case file so that we can create new dataframe with child and parent column

        Parameters
        ----------
        x : .

        Returns
        -------
        new_x : .

        """
        new_x = []
        pre = ""
        for e in x:
            if pre == "":
                new_x.append(pre + e)
                pre = pre + e
            else:
                new_x.append(pre + "_" + e)
                pre = pre + "_" + e
        return new_x

    def get_child_parent_list(self, x):
        """
        Helper function

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        xx = x[::-1]
        for i in range(len(xx) - 1):
            self.child_parent_list.append([xx[i], xx[i + 1]])

    def get_child_parent_df(self, df):
        """


        Parameters
        ----------
        df : CASE standards dataframe .

        Returns
        -------
        None.
        But stores child_parent_df in object which is useful in creating tree structure

        """
        df["smartLevel"] = df["smartLevel"].apply(lambda x: str(x))
        df["smartLevel_LIST"] = df["smartLevel"].apply(lambda x: x.split("."))
        df["smartLevel_LIST_recode"] = df["smartLevel_LIST"].apply(
            lambda x: self.smart_level_recode(x)
        )
        df["smartLevel_unique"] = df["smartLevel"].apply(
            lambda x: "_".join(x.split("."))
        )
        _ = df["smartLevel_LIST_recode"].apply(lambda x: self.get_child_parent_list(x))
        self.child_parent_df = pd.DataFrame(
            self.child_parent_list, columns=["Child", "Parent"]
        )
        self.child_parent_df = self.child_parent_df.drop_duplicates(
            keep="first"
        ).reset_index(drop=True)
