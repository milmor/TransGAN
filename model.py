'''TransGAN model for Tensorflow.

# Reference paper
- Yifan Jiang, Shiyu Chang and Zhangyang Wang. 
  [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](
    https://arxiv.org/abs/2102.07074) 
  
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
    def __init__(self, model_dim, n_heads, initializer='glorot_uniform'):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wk = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wv = layers.Dense(model_dim, kernel_initializer=initializer)

        self.dense = layers.Dense(model_dim, kernel_initializer=initializer)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
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
    def __init__(self, n_patches, model_dim, initializer='glorot_uniform'):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = layers.Embedding(
            input_dim=n_patches, output_dim=model_dim,
            embeddings_initializer=initializer
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)

    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='glorot_uniform'):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(model_dim, n_heads, initializer=initializer)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu', 
                         kernel_initializer=initializer), 
            layers.Dense(model_dim, kernel_initializer=initializer),
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
    def __init__(self, model_dim=1024, noise_dim=256, depth=[5, 4, 2], 
                 heads=[4, 4, 4], mlp_dim=[4096, 1024, 256], initializer='glorot_uniform'):
        super(Generator, self).__init__()
        self.init = tf.keras.Sequential([
            layers.Dense(8 * 8 * model_dim, use_bias=False, 
                         kernel_initializer=initializer),
            layers.Reshape((8 * 8, model_dim))
        ])     
        
        self.pos_emb_8 = PositionalEmbedding(64, model_dim, initializer=initializer)
        self.block_8 = tf.keras.Sequential()
        for _ in range(depth[0]):
            self.block_8.add(TransformerBlock(model_dim, heads[0], mlp_dim[0], 
                                              initializer=initializer))
         
        self.pos_emb_16 = PositionalEmbedding(256, model_dim // 4, initializer=initializer)
        self.block_16 = tf.keras.Sequential()
        for _ in range(depth[1]):
            self.block_16.add(TransformerBlock(model_dim // 4, heads[1], mlp_dim[1], 
                                               initializer=initializer))
            
        self.pos_emb_32 = PositionalEmbedding(1024, model_dim // 16, initializer=initializer)
        self.block_32 = tf.keras.Sequential()
        for _ in range(depth[2]):
            self.block_32.add(TransformerBlock(model_dim // 16, heads[2], mlp_dim[2], 
                                               initializer=initializer))

        self.ch_conv = layers.Conv2D(3, 3, padding='same', kernel_initializer=initializer)
                       
    def call(self, z):
        B = z.shape[0]
        x = normalize_2nd_moment(z)
   
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
                 heads=[4, 4, 4], mlp_dim=[768, 1536, 1536], img_size=32, 
                 policy='color,translation,cutout', initializer='glorot_uniform'):
        super(Discriminator, self).__init__()
        '''Encode image'''
        patches_32 = (img_size // patch_size)**2
        self.patch_32 = tf.keras.Sequential([
            layers.Conv2D(model_dim[0], kernel_size=patch_size, 
                          strides=patch_size, padding='same', kernel_initializer=initializer)
        ])
        self.pos_emb_32 = PositionalEmbedding(n_patches=patches_32, 
                                              model_dim=model_dim[0], initializer=initializer)
        self.block_32 = tf.keras.Sequential()
        for _ in range(depth[0]):
            self.block_32.add(TransformerBlock(model_dim[0], heads[0], mlp_dim[0], 
                                               initializer=initializer))

        patches_16 = ((img_size//2) // patch_size)**2
        self.patch_16 = tf.keras.Sequential([
            layers.Conv2D(model_dim[1], kernel_size=patch_size*2, 
                          strides=patch_size*2, padding='same', kernel_initializer=initializer),
        ])
        self.pos_emb_16 = PositionalEmbedding(n_patches=patches_16, 
                                              model_dim=model_dim[0] + model_dim[1])
        self.block_16 = tf.keras.Sequential()
        for _ in range(depth[1]):
            self.block_16.add(TransformerBlock(model_dim[0] + model_dim[1], 
                                               heads[1], mlp_dim[1], initializer=initializer))
        '''Last block'''
        self.last_block=TransformerBlock(model_dim[0] + model_dim[1], heads[2], mlp_dim[2], 
                                         initializer=initializer)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        '''Encode cls_token'''        
        self.cls_dim = model_dim[0] + model_dim[1]
        self.cls_token = self.add_weight(name='cls_token',
                                         shape=(1, self.cls_dim),
                                         initializer=initializer,
                                         trainable=True)
        '''Logits'''
        self.logits = layers.Dense(1, kernel_initializer=initializer)
        self.policy = policy

    def call(self, img):
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

        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, self.cls_dim])
        x = tf.concat([cls_tokens, x], 1)
        x = self.last_block(x)
        x = self.norm(x)
        return [self.logits(x[:, 0])]
