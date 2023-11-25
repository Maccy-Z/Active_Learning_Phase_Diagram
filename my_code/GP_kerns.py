import torch
import math


class Kernel():
    def __init__(self, param_holder):
        self.param_holder = param_holder

    def kern(self, X1, X2):
        raise NotImplementedError

    def diag_kern(self, X):
        """
        X.shape = (n, dim)
        Diagonal kernel means r=0, so just return scale
        """
        params = self.param_holder.get_params()
        kern_scale, kern_len = params['kern_scale'], params['kern_len']

        # Matérn 1/2 Kernel formula
        diag = kern_scale * torch.ones(X.shape[0])
        return diag


class RBF(Kernel):
    def __init__(self, param_holder):
        super().__init__(param_holder)

    def kern_rbf(self, X1, X2):
        """
        X1.shape = (n, dim)
        X2.shape = (m, dim)
        """
        X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
        X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

        params = self.param_holder.get_params()
        kern_scale, kern_len = params['kern_scale'], params['kern_len']

        sqdist = torch.norm(X1 - X2, dim=2, p=2).pow(2)
        k_r = kern_scale * torch.exp(-0.5 * sqdist / kern_len ** 2)

        return k_r


class Matern12(Kernel):
    def __init__(self, param_holder):
        super().__init__(param_holder)

    def kern(self, X1, X2):
        """
        X1.shape = (n, dim)
        X2.shape = (m, dim)
        """
        X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
        X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

        params = self.param_holder.get_params()
        kern_scale, kern_len = params['kern_scale'], params['kern_len']

        # Calculate the pairwise Euclidean distance
        r = torch.norm(X1 - X2, dim=2, p=2)

        # Matérn 1/2 Kernel formula
        k_matern_half = kern_scale * torch.exp(-r / kern_len)

        return k_matern_half


class Matern32(Kernel):
    def __init__(self, param_holder):
        super().__init__(param_holder)

    def kern(self, X1, X2):
        """
        X1.shape = (n, dim)
        X2.shape = (m, dim)
        """
        X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
        X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

        params = self.param_holder.get_params()
        kern_scale, kern_len = params['kern_scale'], params['kern_len']

        # Calculate the pairwise Euclidean distance
        r = torch.norm(X1 - X2, dim=2, p=2)

        # Matérn 3/2 Kernel formula
        sqrt_3_r_l = math.sqrt(3) * r / kern_len
        k_matern = kern_scale * (1 + sqrt_3_r_l) * torch.exp(-sqrt_3_r_l)

        return k_matern


class Matern52(Kernel):
    def __init__(self, param_holder):
        super().__init__(param_holder)

    def kern(self, X1, X2):
        """
        X1.shape = (n, dim)
        X2.shape = (m, dim)
        """
        X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
        X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

        params = self.param_holder.get_params()
        kern_scale, kern_len = params['kern_scale'], params['kern_len']

        # Calculate the pairwise Euclidean distance
        r = torch.norm(X1 - X2, dim=2, p=2)

        # Matérn 5/2 Kernel formula
        sqrt_5_r_l = math.sqrt(5) * r / kern_len
        k_matern_five_half = kern_scale * (1 + sqrt_5_r_l + 5 * r.pow(2) / (3 * kern_len.pow(2))) * torch.exp(-sqrt_5_r_l)

        return k_matern_five_half
