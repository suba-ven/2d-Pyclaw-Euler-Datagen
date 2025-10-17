from SBIConfig import SBIWarmStartDataConfig

def generate_description(config: SBIWarmStartDataConfig):
    """
    Generate a string description of the simulation configuration.

    Args:
        config (SBIWarmStartDataConfig): Configuration object for the simulation.

    Returns:
        str: Formatted string with all relevant config information.
    """
    desc = []
    desc.append("=== Warm Start Data Generation Config ===")
    
    # Domain and grid
    desc.append(f"Domain: x={config.x_domain}, y={config.y_domain}")
    desc.append(f"Grid resolution: nx={config.nx}, ny={config.ny}")
    
    # Time stepping
    desc.append(f"Simulation time: t_final={config.t_final}, n_steps={config.n_steps}")
    desc.append(f"Observation starts at frame: {config.obs_start_idx}")
    
    # Physical parameters
    desc.append("Physical Parameters:")
    desc.append(f"  gamma={config.gamma}")
    desc.append(f"  x0={config.x0}, y0={config.y0}, r0={config.r0}")
    desc.append(f"  rhoin={config.rhoin}, pin={config.pin}")
    desc.append(f"  rhoout={config.rhoout}, pout={config.pout}")
    desc.append(f"  pinf={config.pinf}, xshock={config.xshock}")
    
    # Thresholding and filtering
    desc.append("Numerical Parameters:")
    #desc.append(f"  threshold={config.threshold}, ll_eps={config.ll_eps}")
    #desc.append(f"  gaussian_filter_sigma={config.gaussian_filter_sigma}")
    
    # Parameter inference
    desc.append("Uncertain Parameters:")
    for param, bounds in zip(config.state_param_list, config.state_param_bounds):
        desc.append(f"  {param}: range = [{bounds[0]}, {bounds[1]}]")

    return "\n".join(desc)

