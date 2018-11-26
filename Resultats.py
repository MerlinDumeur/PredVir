import pandas as pd


class Resultats:

    def __init__(self,ds,hash,results_dict):

        try:

            self.df = Resultats.from_file(ds)

        except FileNotFoundError:

            self.df = create_df(hash,ds)

    def from_file(ds):

        filename = ds.get_foldername() + '.pkl'
        return pd.read_pickle(filename)

    def create_df(ds,results_dict,hash):

        Index = 
        return pd.DataFrame(index=*results_dict,)

    def save_data(self,hash,results_dict):

        Index = 