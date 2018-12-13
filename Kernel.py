class Kernel:

    def __init__(self,k_fun,k_args):

        self.fun = k_fun
        self.args = k_args

    def compute(self,X,Y,**k_args):

        return self.fun(X,Y,**self.args)

    def get_params(self):

        param_dict = {k:v for k,v in self.args.items()}
        param_dict['Kernel'] = self.fun.__name__

        return param_dict
