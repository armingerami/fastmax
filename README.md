Dependencies: pytorch and math.

How to use: The provided "fast_multihead_attention" function should be used to replace the multihead attention block in your transformers. 
For example, if your code is o = multiheadattention(q,k,v), where o is the result of softmax(q.k^T)*v, it should be replaced with o = fast_multihead_attention(q,k,v).
More details such as the optional inputs are explained in "fast_multihead_attention.py".
