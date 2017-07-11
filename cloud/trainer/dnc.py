import collections
import tensorflow as tf
import numpy as np 

import controller as contr
import access
import sonnet as snt


DNCState = collections.namedtuple('DNCState' , ('access_output','access_state' , 'controller_state' ) )


class DNC( snt.RNNCore ):

    def __init__(self , access_config , controller_config ,  output_size , name = "my-dnc" ):


        super(DNC , self).__init__( name = name )
        # create the controller

        with self._enter_variable_scope():
            self._controller = contr.RnnInstacart(**controller_config )
            self._access = access.MemoryAccess(**access_config)


        self._access_output_size = np.prod( self._access.output_size.as_list() )
        self._output_size = output_size 
        
        self._output_size = tf.TensorShape( [ output_size ] )
        
        #self._output_size = tf.TensorShape([output_size])
        self._state_size =  DNCState(
        access_output= self._access_output_size ,
        access_state= self._access.state_size ,
        controller_state=  self._controller.state_size 
        )
        

    def clip_output(self , output ):

        return tf.clip_by_value(output , clip_value_min = 0 , clip_value_max = 1.0  )

    def _build(self , inputs , prev_state ):

        # inputs are [ LEN , batch_size , TOT]
        # prev_state for the network
        # prev_state.controller_state # is needed
          
        batch_flatten = snt.BatchFlatten()
        """
        prev_controller_state = prev_state.controller_state
        prev_access_output = prev_state.access_output 

        
        controller_input = tf.concat([batch_flatten( inputs ) , batch_flatten(  prev_access_output    )] , 1 )
        """
        print("dnc shape")
        print( inputs.shape )

        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        ## waithhhhh
        controller_inputs = tf.concat(
            [batch_flatten(inputs) , batch_flatten(prev_access_output) ]  , 1 ) 
        
        controller_output , controller_state = self._controller( controller_inputs , prev_controller_state  )
        
        access_output , access_state = self._access( controller_output , prev_access_state )
        
        ## TODO ADD LINEAR LAYER 
        output = tf.concat( [ controller_output , batch_flatten( access_output ) ] , 1 )
        print("wash goin on ")
        print( access_output.shape  )
        print( controller_output.shape )


        output = snt.Linear( output_size = self._output_size.as_list()[0] , name = "linear_output" )(output)

        output = self.clip_output( output )
        
        return output , DNCState(
            controller_state = controller_state,
            access_state = access_state ,
            access_output =  access_output ,

        ) 
    def initial_state(self , batch_size , dtype = tf.float32  ):

        return DNCState(
            controller_state = self._controller.initial_state(batch_size , dtype) ,
            access_state =  self._access.initial_state(batch_size , dtype) ,
            access_output = tf.zeros( [batch_size] + self._access.output_size.as_list(), dtype)
        )

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size
