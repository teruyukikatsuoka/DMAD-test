import tensorflow as tf

def forward_diffusion_sample(x_0, t, constant_dict, config):
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod= constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod']

    noise = tf.random.normal(shape=x_0.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
    device = config.model.device
    
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape, config
    )

    x = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    x = x.to(device)
    noise = noise.to(device)
    return x, noise


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)