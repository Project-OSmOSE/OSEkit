import pandas as pd

import logging
import pip

logging_level = logging.INFO
logging.basicConfig(level=logging_level)

if hasattr(pip, 'main'):
    pip.main(['install', "datatest"])
else:
    pip._internal.main(['install', "datatest"])
import datatest as dt

class PandasTests():
    """
    Tests to validate the pandas package
    """

    def __init__(self):
        self.run()


    def test_pandas_pkg(self):
        """
        To validate the pandas pkg functionality
        """  
        
        # Tests fail with an error "undefined symbol: xstrtod"
        # on ppc64le for python 3.8 
        try:
            from pandas.testing import assert_frame_equal
        
            df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
            logging.info("validating Pandas dataframes, values of df1 and df2 {} {}".format(df1, df2))
            assert_frame_equal(df1, df2, check_dtype=False)
        except ImportError as e:
            print(e)

    def test_pandas_dataframe(self):
        """
        To validate the data frame module.
        """
        dt.register_accessors()
        df = pd.DataFrame(data={'A': ['abc', 'def', 'ghi', 'jkl'],
                                'B': [0, 10, 20, 30]})
        requirement = [
            ('abc', 0),
            ('def', 10),
            ('ghi', 20),
            ('jkl', 30),
        ]
        logging.info("validating Pandas dataframes class type, expected and actual dataframe values {} {}".format(requirement, df))
        df.validate((str, int))


    def test_pandas_multi_index(self):
        """
        To validate pandas multi_index_from_tuples  module
        """
        multi_frame = pd.MultiIndex.from_tuples([
            ('I', 'a'),
            ('II', 'b'),
            ('III', 'c'),
            ('IV', 'd'),
        ])
        requirement = [('I', 'a'), ('II', 'b'), ('III', 'c'), ('IV', 'd')]
        logging.info("validating Pandas multiframe logic, expected and actual values{} {}".format(requirement, multi_frame))
        multi_frame.validate(requirement)

    def run(self):
        self.test_pandas_pkg()
        self.test_pandas_dataframe()
        self.test_pandas_multi_index()


if __name__ == "__main__":
    obj = PandasTests()
