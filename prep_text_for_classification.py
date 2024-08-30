import tensorflow as tf

#Define Model inputs
q = "How old is the patient?"
p = '''
The patient is an 84-year-old male named Sebastian. 
He has no history of chronic liver conditions but is 
showing mild degenerative changes and some symptoms related to intermittent ascites.
'''
#Tokenize sentence
tokenizer.tokenize(q)

classification_token = '[CLS]'
separator_token = '[SEP]'

CLS = tokenizer.cls_token
SEP = tokenizer.sep_token

tokens = []
tokens.append(classification_token)
tokens.extend(q_tokens)
tokens.append(separator_token)
tokens.extend(p_tokens)

#Convert Tokens to numerical representation
tokenizer.convert_tokens_to_ids(tokens)

#Apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 60

token_ids = pad_sequences([token_ids], padding="post", maxlen=max_length)

#convert to Tensor
token_ids = tf.convert_to_tensor(token_ids)

#Add the input mask
from tensorflow.keras import layers

masking_layer = layers.Masking()

unmasked = tf.cast(
    tf.tile(tf.expand_dims(tf.convert_to_tensor(
        token_ids), axis=-1), [1, 1, 1]),
    tf.float32)

masked = masking_layer(unmasked)
token_mask = masked._keras_mask

#convert the padded token ids list into a Tensor
padded_token_ids = tf.convert_to_tensor(padded_token_ids)

#reshape the Tensor as needed
padded_token_ids = tf.expand_dims(padded_token_ids, 0)

#check the match
padded_token_ids.shape == token_ids.shape
