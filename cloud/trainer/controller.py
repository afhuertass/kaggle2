

import tensorflow as tf
import sonnet as snt

tf.flags.DEFINE_integer( "num_training_steps" , 10000 , ''  )

tf.flags.DEFINE_integer( "report_interval" , 100  , '' )

tf.flags.DEFINE_integer( "reduce_learning_interval" , 200 ,''    )

tf.flags.DEFINE_integer( "lstm_depht" , 2  ,'' )
tf.flags.DEFINE_integer( "lstm_units" , 50  , '' )

tf.flags.DEFINE_integer( "lenght_features" , 100 , ''  )

tf.flags.DEFINE_float( "learning_rate" , 0.001  , '' )
tf.flags.DEFINE_float( "learning_reduce_multi" , 0.1 , ''   )
tf.flags.DEFINE_float( "opti_epsilon" , 0.001  , '' )


tf.flags.DEFINE_string( "checkpoint_dir" , "./train/dir" , ''   )
tf.flags.DEFINE_integer( "checkpoint_interval" , 100  , ''  )



class RnnInstacart(snt.AbstractModule ):

    def __init__( self , num_hidden  , depth ,  output_size , use_skip_connections = True   , use_dynamic = True ,name = "rnn_instacart" ):
        self.num_hidden = num_hidden 
        self.depth = depth
        self.use_skip_connections =  use_skip_connections
        self.use_dynamic = use_dynamic
        
        self._output_size = output_size 
        super(RnnInstacart, self).__init__(name=name)
        print("wtf")
        print(num_hidden)
        print(depth)
        with self._enter_variable_scope():
            # layer of lstm units
            self._output_module = snt.Linear( self._output_size , name = "output" )
            self._lstms = [ snt.LSTM( num_hidden , name="lstm_{}".format(i) ) for i in range(depth )  ]

            self._core = snt.DeepRNN( self._lstms , skip_connections = self.use_skip_connections , name = "deep_lstm"
            )

            
            # create some layers here
            
        return 

    
    
    def _build(self , inputs_sequence , prev_state ):

        # input_sequence [ LEN , batch_size , output_size ]
        input_shape = inputs_sequence.get_shape()
        
        batch_size =  input_shape[0]
        
        #initial_state = self._core.initial_state( batch_size )
        print( "controller shape ")
        print( inputs_sequence.shape )
        output_seq , final_state = self._core(inputs_sequence , prev_state )

        
        """
        if self.use_dynamic :
            output_seq , final_state = tf.nn.dynamic_rnn(
                cell = self._core ,
                inputs = inputs_sequence ,
                time_major = True ,
                initial_state = initial_state 
            )
        else :
            rnn_input_seq = tf.unstack( input_sequence )
            output , final_state = tf.contrib.rnn.static_rnn(
                cell = self._core  ,
                inputs = inputs_rnn_input_seq ,
                initial_state = initial_state
            )
            output_seq = tf.stack( output )

        # return the output_seq, and final_sate
        
        batch_output_seq_m = snt.BatchApply(self._output_module )

        output_seq = batch_output_seq_m( output_seq )
        """
        print(output_seq.shape )
        
       
        #output_seq = snt.Linear( self._output_size , name="output_linear" )( output_seq )
        
        
        return output_seq , final_state
    
        
    def initial_state(self , batch_size , dtype ):

        return self._core.initial_state(batch_size , dtype )
        
    @property
    def state_size(self):
        return self._core.state_size

    @property
    def output_size(self):
        return self._core._output_size
