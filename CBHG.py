class CBHG(object):
    
    def __init__(self, input_size, T = 40, embed_dim = 256, lr = 1e-4, batch_size = 256):
        '''
        input_size: # of char len(ALPHABET)
        '''
        self.input_size = input_size
        self.batch_size = batch_size
        self.seq_len = T
        self.seqlen = tf.placeholder(dtype = tf.int32, shape = [None])
        self.embed_d = embed_dim
        self.input_x = tf.placeholder(dtype=tf.int32, 
                                      shape =[None, self.seq_len],name = 'input_x')
        self.labels = tf.placeholder(dtype = tf.float32, name = 'labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
        self.loss = tf.constant(0.0)
        self.layers = 0
        self.K = 8
        self.seq_len = tf.placeholder(tf.int32,[None]) 
        
            ## =========== Pre-train =========== ##
        W_emb = tf.get_variable(name = 'W_emb', 
                                initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32),
                                shape = [input_size,self.embed_d], dtype = tf.float32)
        input_x = tf.nn.embedding_lookup(W_emb,self.input_x)
        input_x = tf.expand_dims(input_x,-1) ## b,T,d,1
            
        fc1 = self.dense(input_x, in_dim = self.embed_d,
                         out_dim = self.embed_d, layer = self.layers)
        fc1 = tf.nn.relu(fc1)
        with tf.name_scope('pre-dropout-%i'%self.layers) as scope:
            dropout1 = tf.nn.dropout(fc1,self.dropout_keep_prob)
        self.input = dropout1 #b,T,d,1
        dropout1 = tf.transpose(self.input,[0,2,1,3]) # b,d,T,1
            ## =========== CBHG ============ ##
        self.layers+=1
        conv_list = []
        previous_shape = [d.value for d in dropout1.get_shape().dims]
        for k in range(1,self.K+1):
            conv = self.convolution(dropout1, kh = previous_shape[1], kw = k, filter_in = 1, filter_out = 64,layer = self.layers)
            self.layers += 1
            conv_list.append(tf.squeeze(conv,axis = 1))
        conv0 = tf.concat(conv_list,axis=2) # b,T,filter_out*self.K 
        conv0 = tf.expand_dims(tf.transpose(conv0,[0,2,1]),-1) # b,filter_out*self.K , T, 1
        k_size = [1,1,2,1]
        max_pool = self.pooling(conv0,k_size,strides = [1,1,1,1],layer = self.layers)
        #self.out = max_pool
        self.layers += 1
        pool_shape = [d.value for d in max_pool.get_shape().dims]
        conv1 = self.convolution(max_pool,kh = self.K*64, kw = 3,
                                 filter_in = 1, filter_out = self.embed_d, layer = self.layers)
       
        conv1 = tf.squeeze(conv1,axis = 1) # b,T,d
        
        conv1d_project = tf.expand_dims(conv1,-1)
        resnet = self.input+ conv1d_project # b,T,d = 128,1
        highway_net = self.highway(input_ = resnet, size = [64,64], num_layers=3, f=tf.nn.relu, scope='Highway')
        # b, T, (d = 128) , 1
        ## notice since num_layers of network consecutive, size = [in,out] must be same above
        bi_rnn = self.Bi_RNN(tf.squeeze(highway_net,-1))
        h_last = bi_rnn[:,-1,:]
        h_last = tf.reshape(h_last,[self.batch_size,1,64,1])
        n_in = h_last.get_shape().dims[2].value
        scores = self.dense(h_last,in_dim = n_in, out_dim = 1, layer = self.layers)
        self.scores = tf.squeeze(scores,[-1,-2])
        self.loss += tf.reduce_mean(tf.square(self.scores-self.labels))
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        #grad_var = optimizer.compute_gradients(loss = self.loss,var_list = self.params, aggregation_method = 2)
        #self.train_op = optimizer.apply_gradients(grad_var)
        self.train_op = optimizer.minimize(self.loss)
        
        
        
    def convolution(self,input_x, kh, kw, filter_in, filter_out, layer):
        ## 1D-conv, so kw = D input_dim
        ## kw = k, where k = 1 ... 16
        filter_shape = [kh,kw,filter_in,filter_out]

        with tf.variable_scope("conv-%i-%i" %(layer,kw)) as scope:
                
            W = tf.get_variable(name = 'W-%i'% layer,
                                initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32),
                                shape = filter_shape)
            b = tf.get_variable(name = 'b-%i'% layer,
                                initializer = tf.constant_initializer(0.1),
                                shape = [filter_out], dtype = tf.float32)
            
            ### why not "SAME" work here 
            left = (kw-1)/2
            right = kw - 1 - left
            input_x = tf.pad(input_x,[[0,0],[0,0],[left,right],[0,0]])
            conv = tf.nn.conv2d(input = input_x,
                                filter = W,
                                strides = [1,1,1,1], ## kernel size changes with k, but not stride!!!
                                padding = 'VALID',
                                name = "conv-%i-%i" %(layer,kw))
            h = tf.nn.relu(tf.nn.bias_add(conv,b))
        return h
        
    def pooling(self,input_x,k_size,strides,layer):
        
        with tf.name_scope('Pooling_{0}'.format(layer)) as scope:
            pool = tf.nn.max_pool(value = input_x,ksize = k_size, 
                                  strides = strides,
                                  padding = 'SAME')    
        return pool

    
    def highway(self, input_, size, num_layers=1, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        
        size = [in_dim,out_dim]
        """
        n_in,n_out = size
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                self.layers += 1
                g = f(self.dense(input_, in_dim = n_in, out_dim = n_out, 
                                 layer = self.layers))
                t = tf.sigmoid(self.dense(input_, in_dim = size[0],out_dim = size[1], 
                                     layer = self.layers))
                output = t * g + (1. - t) * input_
                input_ = output

        return output
    
    def dense(self, input_x, in_dim, out_dim, layer, scopes = None):
        '''
        input:
            input_x: (batch, T, D, 1)
            input_size: D*T*1
        output:
            (batch, output_unit, T, 1)
        '''

        x_flat = tf.squeeze(input_x,-1)
        ### if change to variable scope: Then, outside this function,
        ### if do not specify this variable scope, we cannot access it
        with tf.variable_scope(scopes or 'dense-%i'%layer) as scope:
            try:
                matrix = tf.get_variable("Matrix", [in_dim, out_dim], dtype=tf.float32)
                bias_term = tf.get_variable('bias_term',[out_dim],dtype = tf.float32)
            except ValueError:
                scope.reuse_variables()
                matrix = tf.get_variable('Matrix')
                bias_term = tf.get_variable('bias_term')
        matrix_b = tf.tile(tf.expand_dims(matrix,0),[self.batch_size,1,1])
        bias = tf.tile(tf.expand_dims(bias_term,0),[self.batch_size,1])
        bias = tf.expand_dims(bias,-1)
        dense = tf.matmul(x_flat,matrix_b)
        return tf.expand_dims(dense,-1)
    
    def Bi_RNN(self,input_x):
        
        fw_cell = tf.contrib.rnn.LSTMCell(64, forget_bias=1.0, state_is_tuple=True)
        #bw_cell = tf.contrib.rnn.LSTMCell(128, forget_bias=1.0, state_is_tuple=True)
        
        out, states = tf.nn.dynamic_rnn(fw_cell, input_x, self.seq_len, dtype = tf.float32) ## this step does create variables    
#         h, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell,
#                                                      cell_bw = bw_cell,
#                                                      inputs = input_x,
#                                                      sequence_length=self.seqlen,
#                                                      dtype=tf.float32)
#         out = tf.concat(h,axis = 2)
        return out