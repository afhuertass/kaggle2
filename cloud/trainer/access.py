
import adressing
import tensorflow as tf 
import numpy as np
import sonnet as snt
import collections



AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


def _erase_and_write(memory , address ,reset_weights , values):

    with tf.name_scope('erase_memory' , values = [memory,address,reset_weights] ):

        expand_address = tf.expand_dims( address , 3 )
        reset_weights = tf.expand_dims( reset_weights , 2 )
        weighted_resets = expand_address*reset_weights
        reset_gate = tf.reduce_prod( 1 - weighted_resets , [1] )
        memory *=  reset_gate
        print("some shapes")
        print( address.shape )
        print( values.shape)
    with tf.name_scope('additieve_write' , values =[ memory, values ] ):
        add_matrix = tf.matmul( address , values , adjoint_a = True )
        memory += add_matrix

    return memory 


class MemoryAccess(snt.RNNCore ):

    
    def __init__(self , memory_size = 128 , w_size = 20 , num_reads = 1 ,
                 num_writes = 1 , 
                 name="memory_access" ):



        super(MemoryAccess , self).__init__(name=name)
        self._memory_size = memory_size
        self._w_size = w_size
        self._num_reads = num_reads
        self._num_writes = num_writes
    
        self._write_weights_mod = adressing.CosineAttention( num_writes , w_size ,name = "write_content_weights" )
        self._read_weights_mod = adressing.CosineAttention( num_reads , w_size ,name = "read_content_weights")
        

        self._linkage = adressing.Linkage( memory_size , num_writes )
        self._freeness = adressing.Freeness( memory_size  ) 


    def _build(self , inputs , prev_state):

        """
        inputs = tensor [batch_size , input_size ]
        prev_state = instances of access state, with the goal of access previous information about the memory state
        """
        inputs = self._read_inputs( inputs ) 

        usage = self._freeness(
            write_weights = prev_state.write_weights ,
            free_gate = inputs['free_gate'] ,
            read_weights = prev_state.read_weights  ,
            prev_usage = prev_state.usage
        )
        # escribir a la memoria
        write_weights = self._write_weights(inputs , prev_state.memory , usage )
        

        memory = _erase_and_write(
            prev_state.memory ,
            address = write_weights ,
            reset_weights = inputs['erase_vectors'] ,
            values = inputs['write_vectors'] 
        )
        linkage_state = self._linkage( write_weights , prev_state.linkage )

        read_weights = self._read_weights(
            inputs , memory = memory ,
            prev_read_weights = prev_state.read_weights ,
            link = linkage_state.link
        )

        read_words = tf.matmul( read_weights , memory )
        print("access shapes")
        print( read_words.shape )
        return (  read_words , AccessState(
            memory = memory ,
            read_weights = read_weights ,
            write_weights = write_weights ,
            linkage = linkage_state , usage = usage 
        ) )
    
        
    def _read_inputs( self , inputs ):

        # parse inputs to be used
        def _linear( first_dim , second_dim , name , activation= None):

            linear = snt.Linear( first_dim*second_dim , name = name)(inputs)
            if activation is not None:
                linear = activation( linear , name = name+'_activation')

            return tf.reshape(linear , [-1, first_dim , second_dim ])

        # v_t^i
        # por cada write_head existe un vector en R[W] 
        write_vectors = _linear( self._num_writes , self._w_size , 'write_vectors' )

        # por cada write_head hay un vector de erase asociado
        # con la diferencia que estos estan restringidos al intervalo [0,1]
        # por tanto aplicamos una operacion de sigmoide
        erase_vectors = _linear( self._num_writes , self._w_size , 'erase_vectors' , tf.sigmoid)


        # tiene que haber tantas free_gates como read_heads
        # f_t^j  j read_heads
        free_gate = tf.sigmoid( snt.Linear( self._num_reads , name="free_gates") (inputs)  )


        # g_t^{a,i}  determina la alocacion de cada head the escritura, 
        allocation_gate = tf.sigmoid( snt.Linear( self._num_writes , name="allocation_gate" )(inputs) )

        
        
        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = tf.sigmoid( snt.Linear( self._num_writes, name="write_gate" )(inputs) )


        # cada head the lectura tiene 3 modos de lectura b,f, content based
        num_read_modes = 1 + 2 * self._num_writes

        read_mode = snt.BatchApply( tf.nn.softmax)( _linear( self._num_reads , num_read_modes, name="read_modes"  ) )
        # el modo de lectura a aplicar es resultado de una regresion softmax 

        write_keys = _linear( self._num_writes , self._w_size , 'write_keys' )
        write_strengths = snt.Linear( self._num_writes , name = 'write_strenghts')(inputs)

        read_keys = _linear( self._num_reads , self._w_size , 'read_keys' )
        read_strengths = snt.Linear( self._num_reads , name = 'read_strenghts')(inputs)
        
        result = {

            'read_content_keys' : read_keys ,
            'read_content_strengths' : read_strengths ,
            'write_content_keys' : write_keys ,
            'write_content_strengths' : write_strengths ,
            'write_vectors' : write_vectors ,
            'erase_vectors' : erase_vectors ,
            'free_gate' : free_gate ,
            'allocation_gate' : allocation_gate ,
            'write_gate' : write_gate ,
            'read_mode' : read_mode
            

        }

        return result 

    def _write_weights( self ,  inputs  , memory , usage ):

        with tf.name_scope( 'write_weights' , values = [inputs,memory,usage]):
            # cosine attention content based 
            write_content_weights = self._write_weights_mod(
                memory , inputs['write_content_keys'] , inputs['write_content_strengths'] )

            write_allocation_weights = self._freeness.write_allocation_weights(
                usage = usage , write_gates = inputs['allocation_gate']*inputs['write_gate'],
                num_writes = self._num_writes
            )
            
            allocation_gate = tf.expand_dims( inputs['allocation_gate'] , -1 )
            write_gate = tf.expand_dims( inputs['write_gate'] , -1 )

            return write_gate*(allocation_gate*write_allocation_weights  + (1-allocation_gate)*write_content_weights )

    
    def _read_weights(self , inputs , memory , prev_read_weights , link  ):

        with tf.name_scope( 'read_weights' , values = [inputs , memory , prev_read_weights , link   ]  ):

            content_weights = self._read_weights_mod( memory , inputs['read_content_keys'] , inputs['read_content_strengths'] )
            
            forward_weights = self._linkage.directional_read_weights( link , prev_read_weights , forward = True  )
            backward_weights = self._linkage.directional_read_weights( link , prev_read_weights , forward =False  )
            
            backward_mode = inputs['read_mode'][: , : ,  :self._num_writes]
            forward_mode = inputs['read_mode'][: , : , self._num_writes:2*self._num_writes ]
            content_mode = inputs['read_mode'][: , : , 2*self._num_writes ]
            
            
            read_weights = (
                tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(
                    tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
                tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))

            return read_weights

   
    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._w_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)
    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._num_reads, self._w_size])
