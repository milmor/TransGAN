'''TransGAN model for Tensorflow.

# Reference paper
- Yifan Jiang, Shiyu Chang and Zhangyang Wang. 
  [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up]
  (https://arxiv.org/abs/2102.07074). 

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2021
'''
import tensorflow as tf
from tensorflow.keras import layers
from diffaug import DiffAugment


def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = tf.reshape(x, (-1, H, W, C))
    x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
    B, H, W, C = x.shape
    x = tf.reshape(x, (-1, H * W, C))
    return x, H, W

def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)

def scaled_dot_product(q, k, v):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    attn_weights = tf.nn.softmax(scaled_qk, axis=-1)  
    output = tf.matmul(attn_weights, v) 
    return output
    

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim)
        self.wk = layers.Dense(model_dim)
        self.wv = layers.Dense(model_dim)

        self.dense = layers.Dense(model_dim)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_into_heads(q, batch_size)  
        k = self.split_into_heads(k, batch_size)  
        v = self.split_into_heads(v, batch_size)  

        scaled_attention = scaled_dot_product(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim)) 

        output = self.dense(original_size_attention) 
        return output
        

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_patches, model_dim):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = layers.Embedding(
            input_dim=n_patches, output_dim=model_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)

    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, rate=0.0, eps=1e-6):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(model_dim, n_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'), 
            layers.Dense(model_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=eps)
        self.norm2 = layers.LayerNormalization(epsilon=eps)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training):
        x_norm1 = self.norm1(inputs)
        attn_output = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output, training=training) 
        
        x_norm2 = self.norm2(attn_output)
        mlp_output = self.mlp(x_norm2)
        return attn_output + self.drop2(mlp_output, training=training)
        
        
class Generator(tf.keras.models.Model):
    def __init__(self, model_dim=1024, noise_dim=128, depth=[5, 4, 2], 
                 heads=[2, 2, 2], mlp_dim=[512, 512, 512], n_classes=10):
        super(Generator, self).__init__()
        self.class_emb = layers.Embedding(n_classes, noise_dim)
        self.init = tf.keras.Sequential([
            layers.Dense(8 * 8 * model_dim, use_bias=False),
            layers.Reshape((8 * 8, model_dim))
        ])     
        
        self.pos_emb_8 = PositionalEmbedding(64, model_dim)
        self.block_8 = tf.keras.Sequential()
        for _ in range(depth[0]):
            self.block_8.add(TransformerBlock(model_dim, heads[0], mlp_dim[0]))
         
        self.pos_emb_16 = PositionalEmbedding(256, model_dim // 4)
        self.block_16 = tf.keras.Sequential()
        for _ in range(depth[1]):
            self.block_16.add(TransformerBlock(model_dim // 4, heads[1], mlp_dim[1]))
            
        self.pos_emb_32 = PositionalEmbedding(1024, model_dim // 16)
        self.block_32 = tf.keras.Sequential()
        for _ in range(depth[2]):
            self.block_32.add(TransformerBlock(model_dim // 16, heads[2], mlp_dim[2]))

        self.ch_conv = layers.Conv2D(3, 3, padding='same')
                       
    def call(self, x, z):
        B = x.shape[0]
        x = self.class_emb(x) # (B, 1, noise_dim)
        x = tf.squeeze(x, axis=1) # (B, noise_dim)
        z = normalize_2nd_moment(z)
        x *= z
        
        x = self.init(x)
        x = self.pos_emb_8(x)
        x = self.block_8(x)
        x, H, W = pixel_upsample(x, 8, 8)
        
        x = self.pos_emb_16(x)
        x = self.block_16(x)
        x, H, W = pixel_upsample(x, H, W)
        
        x = self.pos_emb_32(x)
        x = self.block_32(x)

        x = tf.reshape(x, [B, H, W, -1])
        return [self.ch_conv(x)]


class Discriminator(tf.keras.models.Model):
    def __init__(self, model_dim=[192, 192], depth=[3, 3], patch_size=2, 
                 heads=[2, 2, 2], mlp_dim=[512, 512, 512], img_size=32, n_classes=10):
        super(Discriminator, self).__init__()
        '''Encode image'''
        patches_32 = (img_size // patch_size)**2
        self.patch_32 = tf.keras.Sequential([
            layers.Conv2D(model_dim[0], kernel_size=patch_size, 
                          strides=patch_size, padding='same')
        ])
        self.pos_emb_32 = PositionalEmbedding(n_patches=patches_32, 
                                              model_dim=model_dim[0])
        self.block_32 = tf.keras.Sequential()
        for _ in range(depth[0]):
            self.block_32.add(TransformerBlock(model_dim[0], heads[0], mlp_dim[0]))

        patches_16 = ((img_size//2) // patch_size)**2
        self.patch_16 = tf.keras.Sequential([
            layers.Conv2D(model_dim[1], kernel_size=patch_size*2, 
                          strides=patch_size*2, padding='same'),
        ])
        self.pos_emb_16 = PositionalEmbedding(n_patches=patches_16, 
                                              model_dim=model_dim[0] + model_dim[1])
        self.block_16 = tf.keras.Sequential()
        for _ in range(depth[1]):
            self.block_16.add(TransformerBlock(model_dim[0] + model_dim[1], 
                                               heads[1], mlp_dim[1]))
        '''Last block'''
        self.last_block=TransformerBlock(model_dim[0] + model_dim[1], heads[2], mlp_dim[2])
        self.norm = layers.LayerNormalization()
        '''Encode cls_token'''        
        self.cls_token = layers.Embedding(n_classes, model_dim[0] + model_dim[1])
        '''Logits'''
        self.logits = layers.Dense(1)
        self.policy = 'color,translation,cutout' 

    def call(self, img, cls):
        img = DiffAugment(img, self.policy)
        x1 = self.patch_32(img)
        B, H, W, C = x1.shape
        x1 = tf.reshape(x1, [B, H * W, C]) 
        x1 = self.pos_emb_32(x1)
        x1 = self.block_32(x1)
        x1 = tf.reshape(x1, [B, H, W, -1]) 
        x1 = tf.nn.avg_pool2d(x1, [1,2,2,1], [1,2,2,1], 'SAME') 
        
        x2 = self.patch_16(img)
        B, H, W, C = x2.shape
        x2 = tf.reshape(x2, [B, H * W, C]) 
        x1 = tf.reshape(x1, [B, H * W, -1]) 
        x = tf.concat([x1, x2], -1)
        x = self.pos_emb_16(x)
        x = self.block_16(x)

        cls_code = self.cls_token(cls)
        x = tf.concat([cls_code, x], 1)
        x = self.last_block(x)
        x = self.norm(x)
        return [self.logits(x[:, 0])]
