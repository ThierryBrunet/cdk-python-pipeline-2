# -*- coding: utf-8 -*-

"""
@author: yogen.chaudhari@powerschool.com
@proprietary: PowerSchool Group LLC

"""
from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self):
        """
        This constructor method helps to access pre-trained model via ST_model variable

        Returns
        -------
        None.

        """

        self.ST_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def get_st_embeddings(self, s):
        """
        This function returns

        Parameters
        ----------
        s : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.ST_model.encode(s.lower())
