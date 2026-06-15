import torch
import torch.nn as nn
import torch.nn.functional as F

def latent_g_eed(symmetry1,symmetry2,model,data,layer_name,num_samples=100,distance='cosine sim'):
    """
    This computes the latent G-empirical equivariance deviation (G-EED) for a 
    specific symmetry G, model f, dataset D, layer name, and distance function. 
    Latent G-EED estimates the extent to which a model deviate from being equivariant
    to user specified actions of G on X and on the hidden activations of the model at
    a chosen layer i.

    Specifically, let G be a finite group and f:X -> Y a deep network where G acts on
    elements of X. Write f_i:X -> X_i for the first i layers of the model so that f_i
    is a function that produces the hidden activations at layer i. We assume that G also
    acts on X_i (even if trivially). To calculate the G-EED we compute

    G-EED := (1/(|D||G|)) sum_{g in G, x in D} m(f_i(gx),gf_i'(x)).

    f_i' is equal to f_i in the case where the action of G on X is faithful and is equal to 
    the centroid of all f_i(gx) as g varies when G acts trivially on X_i.

    Args:
        symmetry1 (TAGTorch symmetry class): The symmetry to be applied to the input data.
        symmetry2 (TAGTorch symmetry class): The symmetry to be applied to the latent space.
        model (Pyorch model): The model whose equivariance we will evaluate
        data (Tensor): G-EED is calculated over a dataset.
        layer_name (nn.Module): The PyTorch layer/module that latent representations should be extracted
        from (e.g., model.fc1).
        num_samples (int, optional): Number of samples to use for orbit computation. Default: 100.
        distance (str, optional): The distance function used to compare latent representations
        to a distance function specific centroid. Options: 'cosine sim', 'euclidean'. Default: 'cosine sim'.

    Returns:
        G-EED (float): The value of the G-EED.

    Example:
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)  # first hidden layer
                self.fc2 = nn.Linear(hidden_dim, hidden_dim) # second hidden layer (optional)
                self.out = nn.Linear(hidden_dim, output_dim) # output layer
        
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.out(x)
                return x
        
        model = MLP(input_dim=100, hidden_dim=64, output_dim=10)
        
        data = torch.randn(10,100)
        
        reversal = vector_reversal()
        trivial = trivial_symmetry()
        
        latent_g_eed_value = latent_g_eed(reversal,trivial,model,data,model.fc1)
        print(latent_g_eed_value)
        latent_g_eed_value = latent_g_eed(reversal,trivial,model,data,model.fc1,distance='euclidean')
        print(latent_g_eed_value)
        
        latent_g_eed_value = latent_g_eed(reversal,reversal,model,data,model.fc1)
        print(latent_g_eed_value)
        latent_g_eed_value = latent_g_eed(reversal,reversal,model,data,model.fc1,distance='euclidean')
        print(latent_g_eed_value)
    """
    
    # Set a hook so that hidden activations can be extracted from a user chosen layer
    activations = {}
    def hook_fn(module, input, output):
        activations['features'] = output
    handle = layer_name.register_forward_hook(hook_fn)

    # List to store activations corresponding to the group acting on the 
    # input space
    orbit_input_action_lst = []
    
    # Iterate through the group elements with each, act on the input 
    # data, store the corresponding hidden activations
    orbit_input_lst = symmetry1.orbit(data,num_samples=num_samples)
    for i in orbit_input_lst:
        model(i)
        orbit_input_action_lst.append(activations['features'].detach().cpu())
    
    orbit_tensor = torch.stack(orbit_input_action_lst)

    # If the action of the group on hidden activations is trivial
    # (this corresponds to invariance) then calculate centroid and 
    # compare each element computed earlier to this using either 
    # Euclidean distance or cosine similiarity
    if symmetry2.trivial:

        # Calculate centroids
        if symmetry1.finite:   
            orbit_centroids = (1/symmetry1.order)*orbit_tensor.sum(dim=0)
        else:
            orbit_centroids = (1/num_samples)*orbit_tensor.sum(dim=0)

        if distance == 'cosine sim':
            # Cosine similarities between orbit elements and centroids
            # Use 1 - similarity to measure deviation (higher = more deviation)
            orbit_sims = F.cosine_similarity(orbit_tensor, orbit_centroids, dim=-1)
            g_eed = (1/orbit_sims.numel())*(torch.sum(1 - orbit_sims)).item()
        elif distance == 'euclidean':
            # Euclidean distances between orbit elements and centroids
            orbit_diffs = orbit_tensor - orbit_centroids
            l2_norms = torch.norm(orbit_diffs, p=2, dim=-1)
            g_eed = (1/l2_norms.numel())*(torch.sum(l2_norms)).item()
    # If the action of the group on hidden activations is nontrivial
    # then calculate the orbit of the action of the group on the 
    # hidden activations 
    else:
        # Take the hidden activation corresponding to the original data 
        # as the representative that we will take the orbit over
        latent_rep = orbit_tensor[0,:]
        # Calculate orbit over all elements of the group
        latent_orbit_lst = symmetry2.orbit(latent_rep,num_samples=num_samples)
        latent_orbit_lst = [i.detach().cpu() for i in latent_orbit_lst]

        latent_orbit_tensor = torch.stack(latent_orbit_lst)

        # Calculate differences pairwise (compare f_i(gx) to gf_i(x))
        if distance == 'cosine sim':
            # Use 1 - similarity to measure deviation (higher = more deviation)
            orbit_sims = F.cosine_similarity(orbit_tensor, latent_orbit_tensor, dim=-1)
            g_eed = (1/orbit_sims.numel())*(torch.sum(1 - orbit_sims)).item()
        elif distance == 'euclidean':
            orbit_diffs = orbit_tensor - latent_orbit_tensor
            l2_norms = torch.norm(orbit_diffs, p=2, dim=-1)
            g_eed = (1/l2_norms.numel())*(torch.sum(l2_norms)).item()

    # Clean up hook to prevent memory leaks
    handle.remove()

    return g_eed

def softmax_g_eed(symmetry,model,data,num_samples=100,distance='kl-divergence',apply_softmax=True):

    """
    This computes the softmax G-empirical equivariance deviation (G-EED) for a 
    specific symmetry G, model f, dataset D, and distance function. 
    Softmax G-EED estimates the extent to which a classifier deviates from being invariant
    to user specified actions of G on X.

    Specifically, let G be a finite group and f:X -> Y a classifier where G acts on
    elements of X. To calculate the G-EED we compute

    G-EED := (1/(|D||G|)) sum_{g in G, x in D} m(f(gx),f'(x)).

    f'(x) is the arithmetic average of f(gx) over all g in G.

    Args:
        symmetry (TAGTorch symmetry class): The symmetry to be applied to the input data.
        model (Pyorch model): The model whose invariance we will evaluate
        data (Tensor): softmax G-EED is calculated over a dataset.
        num_samples (int, optional): Number of samples to use for orbit computation. Default: 100.
        distance (str, optional): The distance function used to compare output distribution predictions.
        Options: 'kl-divergence', 'cosine sim', 'euclidean'. Default: 'kl-divergence'.
        apply_softmax (Boolean, optional): Whether softmax should be computed on model output (it is
        common in pytorch models to compute softmax in the loss function). If softmax is already included
        in the definition of the model, set this to False. Default: True.

    Returns:
        G-EED (float): The value of the G-EED.

    Example:
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)  # first hidden layer
                self.fc2 = nn.Linear(hidden_dim, hidden_dim) # second hidden layer (optional)
                self.out = nn.Linear(hidden_dim, output_dim) # output layer
        
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.out(x)
                return x
        
        model = MLP(input_dim=100, hidden_dim=64, output_dim=10)
        
        data = torch.randn(10,100)
        
        reversal = symmetries.vector_reversal()
        trivial = symmetries.trivial_symmetry()
        
        softmax_g_eed_value = softmax_g_eed(reversal,model,data,distance='kl-divergence')
        print(softmax_g_eed_value)
        softmax_g_eed_value = softmax_g_eed(reversal,model,data,distance='cosine sim')
        print(softmax_g_eed_value)
        softmax_g_eed_value = softmax_g_eed(reversal,model,data,distance='euclidean')
        print(softmax_g_eed_value)
    """

    # Iterate through the group elements with each, act on the input
    # data, store the corresponding output. If necessary, apply softmax
    # to the output.
    orbit_lst = symmetry.orbit(data,num_samples=num_samples)
    orbit_lst = [model(i) for i in orbit_lst]
    if apply_softmax:
        orbit_lst = [F.softmax(i,dim=-1) for i in orbit_lst]
    orbit_lst = [i.detach().cpu() for i in orbit_lst]

    orbit_tensor = torch.stack(orbit_lst) 

    # Calculate centroids
    orbit_centroids = (1/orbit_tensor.shape[0])*orbit_tensor.sum(dim=0)
    # Create a copy of each centroid to compare to orbit elements
    orbit_centroids = orbit_centroids.unsqueeze(0)
    orbit_centroids = orbit_centroids.expand(orbit_tensor.shape[0], -1, -1)
    if distance == 'cosine sim':
        # Cosine similarities between orbit elements and centroids
        # Use 1 - similarity to measure deviation (higher = more deviation)
        orbit_sims = F.cosine_similarity(orbit_tensor, orbit_centroids, dim=-1)
        g_eed = (1/orbit_sims.numel())*(torch.sum(1 - orbit_sims)).item()
    elif distance == 'euclidean':
        # Euclidean distances between orbit elements and centroids
        orbit_diffs = orbit_tensor - orbit_centroids
        l2_norms = torch.norm(orbit_diffs, p=2, dim=-1)
        g_eed = (1/l2_norms.numel())*(torch.sum(l2_norms)).item()
    elif distance == 'kl-divergence':
        log_centroid_prob = torch.log(orbit_centroids.clamp_min(1e-12))
        log_centroid_prob = log_centroid_prob.view(-1, log_centroid_prob.shape[-1])
        orbit_tensor = orbit_tensor.view(-1, orbit_tensor.shape[-1])
        kl_per_column = F.kl_div(log_centroid_prob,orbit_tensor,reduction="none").sum(dim=1)
        g_eed = (1/kl_per_column.numel())*(torch.sum(kl_per_column)).item()
    
    return g_eed